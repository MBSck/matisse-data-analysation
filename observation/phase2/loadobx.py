'''Load an OBX file into a new OB.'''

# Copyright (c) 2018, Michael Pruemm
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Changed by Marten Scheuck in May 2022 to adapt it to the MATISSE
templates"""

import argparse
import sys
import os
import re
import getpass
import tempfile
import p2api

from glob import glob
from contextlib import closing

__all__ = ['readsection', 'login', 'findrun', 'createob', 'addtemplate',
           'absolutetcs', 'siderealtcs', 'findingchart', 'ephemerisfile']


# --- parsing OBX files

def readsection(f):
    '''Read the next section from obxfile f.
    The section ends on the with the next key ending with TEMPLATE.NAME.
    Returns the section as a dictionary and the name of the following
    template. If there is no template, the template name will be None.'''
    section = {}
    while True:
        key, value = readline(f)
        if key is None or key.endswith('TEMPLATE.NAME'):
            break
        section[key] = value
    return section, value


def readline(f):
    '''Read a single key, value pair from obxfile f.'''
    line = f.readline()
    while line:
        line = line.strip()
        if line:
            return parseline(line)
        line = f.readline()
    return None, None  # reached EOF


def parseline(line):
    '''Extract key and value from a line.
    >>> parseline('name "my ob"')
    ('name', 'my ob')
    >>> parseline('parser.unescape "line 1\\nline 2\\n'
    ...           '\\t\\"quoted\\" and backslash \\\\"')
    ('parser.unescape', 'line 1\\nline 2\\n\\t"quoted" and backslash \\\\')
    '''
    key, value = line.split(None, 1)
    if value[0] == value[-1] == '"':  # strip surrounding quotes
        value = value[1:-1]
        # expand escape sequences
        for old, new in [('\\t', '\t'), ('\\n', '\n'), ('\\"', '"'),
                         ('\\\\', '\\')]:
            value = value.replace(old, new)
    return key, value


# --- converting OB properties from OBX to p2 format

def setfields(apiobj, obxdata, mapping):
    '''Set all fields in apiobj from obxdata using mapping.
    >>> target = {'name': '', 'ra': '00:00:00', 'differentialRa': 0}
    >>> obx = {'TARGET.NAME': 'M 32', 'ra': '00:42:41.820',
    ...        'diff_ra': 1, 'diffDec': 2}
    >>> setfields(target, obx, targetMapping)
    >>> [(k, target[k]) for k in sorted(target.keys())]
    [('differentialRa', 0), ('name', 'M 32'), ('ra', '00:42:41.820')]
    '''
    for apiname, obxname, convert in mapping:  # mappings are below
        if apiname in apiobj and obxname in obxdata:
            apiobj[apiname] = convert(obxdata[obxname])

obMapping = [
    ('name', 'name', str),
    ('userPriority', 'userPriority', int),
    ('instrument', 'instrument', str),
]
targetMapping = [
    ('name', 'TARGET.NAME', str),
    ('ra', 'ra', str),
    ('dec', 'dec', str),
    ('properMotionRa', 'propRA', float),
    ('properMotionDec', 'propDec', float),
    ('differentialRa', 'diffRA', float),
    ('differentialDec', 'diffDec', float),
    ('equinox', 'equinox', str),
    ('epoch', 'epoch', float),
]
constraintsMapping = [
    ('name', 'CONSTRAINT.SET.NAME', str),
    ('seeing', 'seeing', float),
    ('skyTransparency', 'sky_transparency', str),
    ('airmass', 'air_mass', float),
    ('fli', 'fractional_lunar_illumination', float),
    ('moonDistance', 'moon_angular_distance', int),
    ('strehlRatio', 'strehlratio', float),
    ('twilight', 'twilight', int),
    ('waterVapour', 'watervapour', float),
    ('atm', 'atm', str),
    ('contrast', 'contrast', float),
    ('baseline', 'baseline', str),
]
obsDescriptionMapping = [
    ('name', 'OBSERVATION.DESCRIPTION.NAME', str),
    ('userComments', 'userComments', str),
    ('instrumentComments', 'InstrumentComments', str),
]

# Change by Marten Scheuck May 2022 - Add MATISSE-Templates


# --- converting template parameters from OBX to p2 format

def convert(obxvalue, typ, currentvalue):
    '''Convert obxvalue (a string) to a value of typ.
    If conversion is not possible, raise ValueError.
    If obxvalue is the special value 'NODEFAULT', keep
    the value in template "as is", i.e. return currentvalue.
    >>> convert('123', 'integer', 99)
    123
    >>> convert('NODEFAULT', 'integer', 99)
    99
    >>> convert('T', 'boolean', False)
    True
    '''
    if obxvalue == 'NODEFAULT':
        return currentvalue  # keep value in template "as is"
    return Converter[typ](obxvalue)


def keywordlistValue(s):
    '''Convert a list of space separated keywords.
    >>> keywordlistValue('value1 value2')
    ['value1', 'value2']
    >>> keywordlistValue('')
    []
    '''
    return s.split()


def intlistValue(s):
    '''Convert a list of space separated integers.
    >>> intlistValue('1 1 2 3 5')
    [1, 1, 2, 3, 5]
    >>> intlistValue('')
    []
    >>> intlistValue('alphanumeric')    # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: invalid literal for int() with base 10: 'alphanumeric'
    '''
    return [int(v) for v in s.split()]


def numlistValue(s):
    '''Convert a list of space separated numbers.
    >>> numlistValue('3.141 2.718')
    [3.141, 2.718]
    >>> numlistValue('')
    []
    >>> numlistValue('alphanumeric')    # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'alphanumeric'
    '''
    return [float(v) for v in s.split()]


def booleanValue(s):
    '''Convert a string into a boolean.
    Invalid values raise ValueError.
    >>> booleanValue('T'), booleanValue('F')
    (True, False)
    >>> booleanValue('1')    # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to boolean: '1'
    '''
    if s == 'T':
        return True
    elif s == 'F':
        return False
    else:
        raise ValueError(f"could not convert string to boolean: '{s}'")


Converter = {
    'string': str,
    'keyword': str,    'keywordlist': keywordlistValue,
    'coord': str,  # no conversion necessary
    'integer': int,    'intlist': intlistValue,
    'number': float,    'numlist': numlistValue,
    'boolean': booleanValue,
    'paramfile': lambda x: x,
    'file': str,  # exported as string
}


# --- parsing time intervals

def parseintervals(timeintervals):
    '''Parse a list of time intervals into a list of triples of strings of the
    form (start, end, x).
    Time intervals are exported as space separated triples which are enclosed
    in curly braces. Tripels are space separated and there is a trailing space.
    >>> parseintervals('{2018-03-27T19:43:00 2018-03-30T23:12:00 0} ')
    [('2018-03-27T19:43:00', '2018-03-30T23:12:00', '0')]
    >>> parseintervals('{36000 43200 1} {21600 22500 0} ')
    [('36000', '43200', '1'), ('21600', '22500', '0')]
    An empty list:
    >>> parseintervals('')
    []
    In this case, the trailing space is missing, therefore the closing brace
    is part of the third value. This third value exists for historical reasons
    and is ignored when importing OBX data.
    >>> parseintervals('{36000 43200 0}')
    [('36000', '43200', '0}')]
    '''
    return [tuple(interval[1:].split())
            for interval in timeintervals.split('} ') if interval]


def stripsecs(v):
    '''Strip seconds off a date-time value.
    >>> stripsecs('2018-03-21T18:45:23')
    '2018-03-21T18:45'
    >>> stripsecs('2018-03-21T19:00')
    '2018-03-21T19:00'
    '''
    return v[:-3] if len(v) > 6 and v[-6] == v[-3] == ':' else v


def hhmm(v):
    '''Convert a number of seconds in [0..86400] into a time in HH:MM format.
    >>> hhmm('0')
    '00:00'
    >>> hhmm(10893)
    '03:01'
    >>> hhmm('86399')
    '23:59'
    >>> hhmm('86400')
    '24:00'
    '''
    minutes = int(v) // 60
    return f'{minutes//60:02d}:{minutes%60:02d}'


# --- using the API

def login(username, password: str = None, server: str = "demo"):
    '''Login to the p2 API with the given username. Return the API connection.

    Parameters
    ----------
    username
    password: str, optional
        If none is given, then it will ask for it
    server: str, optional
        Either 'demo', 'production' for paranal or 'production_lasilla' for la
        silla
    '''
    if password is None:
        prompt = f'Password for {username}: '
        if sys.platform == 'ios':  # assume running in Pythonista
            import console
            password = console.password_alert(prompt)
        elif sys.stdin.isatty():
            password = getpass.getpass(prompt)
        else:
            password = input()
    return p2api.ApiConnection(server, username, password)


def findrun(p2, instrument):
    '''Find the first run using instrument and return its containerId.'''
    myruns, __ = p2.getRuns()
    for r in myruns:
        if r['instrument'] == instrument:
            print('Using {mode} run {progId} {title}'.format(**r))
            return r['containerId']
    else:
        raise RuntimeError(f'There is no run for {instrument}.')


def createob(p2, containerId, filename, obdata):
    '''Create a new OB or CB from obdata in the container.'''
    obname = obdata.get('name', filename)
    itemType = 'OB' if obdata.get('type', 'O') == 'O' else 'CB'
    print(f'Creating {itemType} {obname}')
    ob, version = p2.createItem(itemType, containerId, obname)

    # fill in the OB
    setfields(ob, obdata, obMapping)
    if 'target' in ob:
        setfields(ob['target'], obdata, targetMapping)
    if 'constraints' in ob:
        setfields(ob['constraints'], obdata, constraintsMapping)
    if 'obsDescription' in ob:
        setfields(ob['obsDescription'], obdata, obsDescriptionMapping)

    p2.saveOB(ob, version)
    return ob['obId']


def addtemplate(p2, templatename, templatedata, obId, obxdir):
    '''Add a template to OB or CB obId.'''
    print(f'Creating template {templatename}')
    template, version = p2.createTemplate(obId, templatename)
    templateId = template['templateId']

    # update template parameters
    for p in template['parameters']:
        if p['name'] not in templatedata:
            # no value in export file; leave any template default in place
            continue
        if p['type'] in ('paramfile', 'file'):
            continue  # must be set with explicit API calls; handled below
        p['value'] = convert(templatedata[p['name']], p['type'], p['value'])
    p2.saveTemplate(obId, template, version)

    # upload any paramfile or file parameters
    for p in template['parameters']:
        if p['name'] not in templatedata:
            continue
        if p['type'] not in ('paramfile', 'file'):
            continue
        pname = p['name']
        __, version = p2.getFileParam(obId, templateId, pname, '/dev/null')
        if p['type'] == 'file':  # exported inline
            print(f'Uploading file parameter {pname}')
            # saveFileParam needs a file *name* to open, so create one
            with tempfile.NamedTemporaryFile('w+t', encoding='ascii') as f:
                f.write(templatedata[pname])
                f.file.flush()
                # Does not work on Windows: tempfile cannot be opened again
                p2.saveFileParam(obId, templateId, p['name'], f.name, version)
        else:
            fname = os.path.join(obxdir, templatedata[pname])
            print(f'Uploading paramfile parameter {pname} from {fname}')
            p2.saveFileParam(obId, templateId, p['name'], fname, version)


def absolutetcs(p2, obxtimeintervals, obId):
    '''Set the absolute time constraints on an OB.'''
    tcs = [{'from': stripsecs(start), 'to': stripsecs(end)}
           for start, end, __ in parseintervals(obxtimeintervals)]
    __, version = p2.getAbsoluteTimeConstraints(obId)
    p2.saveAbsoluteTimeConstraints(obId, tcs, version)


def siderealtcs(p2, obxtimeintervals, obId):
    '''Set the sidereal time constraints on an OB.'''
    tcs = [{'from': hhmm(start), 'to': hhmm(end)}
           for start, end, __ in parseintervals(obxtimeintervals)]
    __, version = p2.getSiderealTimeConstraints(obId)
    p2.saveSiderealTimeConstraints(obId, tcs, version)


def loadob(p2, obxfile, containerId=None):
    '''Load obxfile and create an OB in the container.
    If no containerId is given, use the top-level container of the first run
    with a matching instrument.'''

    # file names mentioned in obx file (e.g. finding charts)
    # are relative to the obx file
    obxdir = os.path.dirname(obxfile)

    with open(obxfile, 'r', encoding='ascii', errors='strict') as obx:
        obdata, templatename = readsection(obx)

        if containerId is None:
            containerId = findrun(p2, obdata['instrument'])

        obId = createob(p2, containerId, os.path.basename(obxfile), obdata)

        while templatename:
            templatedata, nexttemplate = readsection(obx)
            addtemplate(p2, templatename, templatedata, obId, obxdir)
            templatename = nexttemplate

    if 'absolute_times_list' in obdata:
        absolutetcs(p2, obdata['absolute_times_list'], obId)

    if 'STTimeIntervals' in obdata:
        siderealtcs(obdata['STTimeIntervals'], obId)

    if 'finding_chart_list' in obdata:
        for fcname in obdata['finding_chart_list'].split():
            filename = os.path.join(obxdir, fcname)
            print(f'Attaching finding chart {filename}')
            p2.addFindingChart(obId, filename)

    if 'ephemeris_file' in obdata:
        filename = os.path.join(obxdir, obdata['ephemeris_file'])
        print(f'Uploading ephemeris file {filename}')
        __, version = p2.getEphemerisFile(obId, '/dev/null')
        p2.saveEphemerisFile(obId, filename, version)

    print('Done.')


def main():
    p = argparse.ArgumentParser(description='Load an OBX file into a new OB.')
    p.add_argument('-u', '--username', dest='username', default='52052',
                   help='user name')
    p.add_argument('obx', help='OBX file to load')
    args = p.parse_args()

    try:
        p2 = login(args.username)
        loadob(p2, args.obx)
    except p2api.P2Error as e:
        code, method, url, error = e.args
        print(error)
    except ValueError as e:
        print(str(e))

if __name__ == '__main__':
    main()

