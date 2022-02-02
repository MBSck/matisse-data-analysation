#!/usr/bin/env python
#
# 2018.09.12     Mark Neeser
#
#                This python script is an adaptation of a shell script to access the ESO science archive
#                enabling the user to:
#
#                1. query the ESO archive
#                  The input parameters include: 
#                   - User Portal login and password (mandatory)
#                   - Archive options:
#                       => target name [optional]
#                       => right ascension & declination [optional]
#                       => programme id [optional]
#                       => instrument [optional]
#                       => start & end dates [optional]
#                       => file Category (CALIB, SCIENCE, ACQUISITION, TECHNICAL, TEST, SIMULATION, or OTHER) [optional]
#                       => file Type (OBJECT, STD, ASTROMETRY, BIAS, DARK, FLAT, SKY, LAMP, DOME, . . .) [optional]
#                       => file Mode (DPR.TECH = IMAGE, SPECTRUM, MOS, IFU, . . .) [optional]
#                       => observation mode (DPR.TYPE = IMAGE, MOS, IFU, SPECTRUM, ECHELLE, MXU, POLARIMETRY, CORONOGRAPHY, INTERFEROMETRY) [optional]
#                       => maximum number of rows to be returned [optional]
#                       => download files (Y/N) [optional]
#
#
#                2. Optionally submit the retrieved/selected files for data download with either:
#                   (a) only the selected files delivered 
#                   (b) the selected files + associated raw calibrations
#                   (c) the selected files + associated processed calibrations
#
#
# 2018.10.18:    Adapted to work with both Python 2 and 3:
#                cp getdata_ESO_archive.py getdata_ESO_archive3.py
#                futurize --stage1 -w getdata_ESO_archive3.py
#
# 2018.12.04:    Changed initial archive query from using "wget" to using "requests.get"
#
# 2018.12.05:    Added a -Mode search (i.e. IMAGE, SPECTRUM, MOS, IFU, etc.)
#
#
#
#
# Call:
#
# getdata_ESO_archive.py -User <user_name> -Password <password> -TargetName <SIMBAD target name> -RA <"hh:mm:ss"> -DEC <"dd:mm:ss"> -Box <search box: "hh:mm:ss"> -RunId <run ID> -Inst <instrument name> -StartDate <start date> -EndDate <end date> -Category <file category: SCIENCE, CALIB, ACQUISITION> -Type <file type: OBJECT, STD, BIAS, etc.> -Mode <file mode: IMAGE, SPECTRUM, MOS, IFU, etc.> -MaxRows <max rows to retrieve> -Download <download files: Y/N> -Deliver <delivery_type: files, files_calib, files_master_calib> <optional: -verbose> <optional to create a record of your query: -record>
# 
# e.g.
#
# getdata_ESO_archive.py -User <user_name> -Password <your_password> -Inst FORS2 -StartDate '2013 01 01' -EndDate '2013 04 01' -Category SCIENCE -Mode IMAGE -MaxRows 30 -Download N -Deliver raw2raw
#
#
# ===================================================================================================
#


from __future__ import print_function
import sys				                           # argv handling 
import os				                           # file system handling 
                                                                   # environment variables 
from time import *                                                 # for dating the queries

import re

import requests





# ==================================================
# These are User configuration parameters to set.
#
# ==================================================


this_prog  = 'getdata_ESO_archive.py' # the name of this program.  For info/warning/error messages

VERSION    = 1.0

url1       = "http://archive.eso.org/wdb/wdb/eso/eso_archive_main/query?"
url2       = "https://dataportal.eso.org/rh/api/requests/"
user_agent = "ESO_RAW_DATA_PROGRAMMATIC_SCRIPT_MJN"

out_format = 'csv'      # output format of archive query



# give the query output tables a unique time stamp:

lt    = localtime()
year, month, day, hour, min, sec = lt[0:6]
ttime = str('%04i-%02i-%02iT%02i:%02i:%02i' %(year, month, day, hour, min, sec))


out_list    = 'output_query_' + ttime + '.' + out_format # output results of the initial query
raw_list    = 'raw_list_'     + ttime + '.ascii'         # raw files contained in the initial query
submit_list = 'submit_'       + ttime + '.ascii'         # submitted request to archive
state_list  = 'state_'        + ttime + '.ascii'         # status of archive request: submitted, error, or completed
down_list   = 'download_'     + ttime + '.sh'            # download shell file from ESO archive
down_dir    = 'download_'     + ttime                    # download directory



# ==========================================
# This is required to strip the raw query
# input string of its nonsence characters.
# ==========================================

def clean_up(text, strip_chars=[], replace_extras={}):
    """
    Remove all occurrences of strip_chars and whitespace at the beginning
    and end of each line, then perform string substitutions specified by
    the replace_extras dictionary (as well as normalizing all whitespace
    to a single space character), and then strip whitespace from the
    beginning and end.

    Any consecutive whitespace is normalized to a single space, but
    you can override these implicit substitutions in replace_extras.

    >>> clean_up(' this is one test\n!\n', replace_extras={'\n': '', 'one': '1'})
    'this is 1 test!'
    """
    # Handle strip_chars
    strip_items = '|'.join(re.escape(s) for s in strip_chars)
    strip_re = r'^(?:{}|\s)+|(?:{}|\s)+$'.format(strip_items, strip_items)
    text = re.sub(strip_re, '', text, re.MULTILINE)

    # Normalize whitespace and handle replace_extras
    replace_keys = list(replace_extras.keys())
    replace_keys.sort(key=len, reverse=True)
    replace_re = '|'.join([re.escape(s) for s in replace_keys] + [r'\s+'])
    return re.sub(
        replace_re,
        lambda match: replace_extras.get(match.group(), ' '),
        text
    ).strip()




# ================================
# read the command line parameters 
# ================================


Narg = len(sys.argv)                    # number of arguments entered by user

if Narg < 3:
	print(this_prog,'[error] insufficient number of arguments given to', this_prog)
	print(this_prog,'should be:')
	print(this_prog, 'getdata_ESO_archive.py -User <user_name> -Password <password> -TargetName <SIMBAD target name> -RA <"hh:mm:ss"> -DEC <"dd:mm:ss"> -Box <search box: "hh:mm:ss"> -RunId <run ID> -Inst <instrument name> -StartDate <start date> -EndDate <end date> -Category <file category: SCIENCE, CALIB, ACQUISITION> -Type <file type: OBJECT, STD, BIAS, etc.> -Mode <file mode: IMAGE, SPECTRUM, MOS, IFU, etc.> -MaxRows <max rows to retrieve> -Download <download files: Y/N> -Deliver <delivery_type: files, files_calib, files_master_calib> <optional: -verbose> <optional to create a record of your query: -record>')
	sys.exit()

for option in sys.argv:
	
	if option == '--version':
		print(this_prog,' Version = ', VERSION)
	if option == '--help':
		print('')
		print(this_prog, 'getdata_ESO_archive.py -User <user_name> -Password <password> -TargetName <SIMBAD target name> -RA <"hh:mm:ss"> -DEC <"dd:mm:ss"> -Box <search box: "hh:mm:ss"> -RunId <run ID> -Inst <instrument name> -StartDate <start date> -EndDate <end date> -Category <file category: SCIENCE, CALIB, ACQUISITION> -Type <file type: OBJECT, STD, BIAS, etc.> -Mode <file mode: IMAGE, SPECTRUM, MOS, IFU, etc.> -MaxRows <max rows to retrieve> -Download <download files: Y/N> -Deliver <delivery_type: files, files_calib, files_master_calib> <optional: -verbose> <optional to create a record of your query: -record>')
		print(' ')
		sys.exit()




# ========================================================================================================================
#
# read the input values and assign them to variables
# Allowable command line parameters include:
#  -User -Password -TargetName -RA -DEC -RunId -Inst -StartDate -EndDate -Type -Category-MaxRows
#
# Adding "-verbose" anywhere in the command line keyword chain will set verbose to true.
# Adding "-record"  anywhere in the command line keyword chain will create an output file record of the archive queries.
#
# ========================================================================================================================

verbose = '-verbose' in sys.argv
record  = '-record'  in sys.argv


try:
    i_user  = sys.argv.index('-User')
except ValueError:
    try:
        i_user  = sys.argv.index('-user')
    except ValueError:
        print(this_prog, '[error] -User <user_name> parameter is missing.')
        print('USAGE: getdata_ESO_archive.py -User <user_name> -Password <password> -TargetName <SIMBAD target name> -RA <"hh:mm:ss"> -DEC <"dd:mm:ss"> -Box <search box: "hh:mm:ss"> -RunId <run ID> -Inst <instrument name> -StartDate <start date> -EndDate <end date> -Category <file category: SCIENCE, CALIB, ACQUISITION> -Type <file type: OBJECT, STD, BIAS, etc.> -Mode <file mode: IMAGE, SPECTRUM, MOS, IFU, etc.> -MaxRows <max rows to retrieve> -Download <download files: Y/N> -Deliver <delivery_type: files, files_calib, files_master_calib> <optional: -verbose> <optional to create a record of your query: -record>')
        print(' ')   
        #print('Note that the maximum number of rows returned by default is 100.')
        print('The search radius around the target name and/or coordinates is, by default, 10 minutes.')
        print('The following example will download directly 32 files (16 fits, 16 nightlog files) onto your disk:')
        print('getdata_ESO_archive.py -User <your username> -Passw <your password> -RunId ''090.C-0733(A)'' -Inst FORS2 -StartDate ''2013 01 01'' -EndDate ''2013 04 01'' -Category SCIENCE -MaxRows 30')
        sys.exit()


try:
    i_psswd  = sys.argv.index('-Password')
except ValueError:
    try:
        i_psswd  = sys.argv.index('-password')
    except ValueError:
        try:
            i_psswd  = sys.argv.index('-Passwd')
        except ValueError:
            try:
                i_psswd  = sys.argv.index('-passwd')
            except ValueError:
                print(this_prog, '[error] -Password <user_password> parameter is missing.')
                print(this_prog, '[error] -User <user_name> parameter is missing.')
                print('USAGE: getdata_ESO_archive.py -User <user_name> -Password <password> -TargetName <SIMBAD target name> -RA <"hh:mm:ss"> -DEC <"dd:mm:ss"> -Box <search box: "hh:mm:ss"> -RunId <run ID> -Inst <instrument name> -StartDate <start date> -EndDate <end date> -Category <file category: SCIENCE, CALIB, ACQUISITION> -Type <file type: OBJECT, STD, BIAS, etc.> -Mode <file mode: IMAGE, SPECTRUM, MOS, IFU, etc.> -MaxRows <max rows to retrieve> -Download <download files: Y/N> -Deliver <delivery_type: files, files_calib, files_master_calib> <optional: -verbose> <optional to create a record of your query: -record>')
                print(' ')   
                print('The search radius around the target name and/or coordinates is, by default, 10 minutes.')
                print('wget must be installed on your machine.')
                print('The following example will download directly 32 files (16 fits, 16 nightlog files) onto your disk:')
                print('getdata_ESO_archive.py -User <your username> -Passw <your password> -RunId ''090.C-0733(A)'' -Inst FORS2 -StartDate ''2013 01 01'' -EndDate ''2013 04 01'' -Category SCIENCE -MaxRows 30')
                sys.exit()

        
try:
    i_target = sys.argv.index('-TargetName')
except ValueError:
    i_target = None

try:
    i_ra = sys.argv.index('-RA')
except ValueError:
    i_ra = None

try:
    i_dec = sys.argv.index('-DEC')
except ValueError:
    i_dec = None

try:
    i_box = sys.argv.index('-Box')
except ValueError:
    i_box = None    

try:
    i_runid = sys.argv.index('-RunId')
except ValueError:
    i_runid = None

try:
    i_inst = sys.argv.index('-Inst')
except ValueError:
    i_inst = None

try:
    i_start = sys.argv.index('-StartDate')
except ValueError:
    i_start = None

try:
    i_end = sys.argv.index('-EndDate')
except ValueError:
    i_end = None

try:
    i_type = sys.argv.index('-Type')
except ValueError:
    i_type = None

try:
    i_cat = sys.argv.index('-Category')
except ValueError:
    i_cat = None

try:
    i_mode = sys.argv.index('-Mode')
except ValueError:
    i_mode = None

try:
    i_max = sys.argv.index('-MaxRows')
except ValueError:
    i_max = None

try:
    i_down = sys.argv.index('-Download')
except ValueError:
    i_down = None

try:
    i_downtype = sys.argv.index('-Deliver')
except ValueError:
    i_downtype = None
    

# Assigning values to the command line input parameters:


user    = sys.argv[i_user+1]
passwd  = sys.argv[i_psswd+1]

if i_target != None:
    target  = sys.argv[i_target+1]
else:
    target  = ''
    
if i_ra != None:
    ra      = sys.argv[i_ra+1]
else:
    ra      = ''

if i_dec != None:
    dec     = sys.argv[i_dec+1]
else:
    dec     = ''
    
if i_box != None:
    box_size = sys.argv[i_box+1]
else:
    box_size = '00:10:00'

if i_runid != None:
    runid   = sys.argv[i_runid+1]
else:
    runid   = ''

if i_inst != None:
    inst    = sys.argv[i_inst+1].upper()                  # instrument name must be capitalised
else:
    inst    = ''

if i_start != None:
    start   = sys.argv[i_start+1]
else:
    start   = ''
   
if i_end != None:
    end     = sys.argv[i_end+1]
else:
    end     = ''

if i_type != None:
    dpr_type = sys.argv[i_type+1]
else:
    dpr_type = ''

if i_mode != None:
    dpr_tech = sys.argv[i_mode+1]
else:
    dpr_tech = ''    

if i_cat != None:
    catg     = sys.argv[i_cat+1]
else:
    catg     = ''

if i_max != None:
    maxr     = sys.argv[i_max+1]
else:
    maxr     = ''

if i_down != None:
    down     = sys.argv[i_down+1]
    if down == 'y' or down == 'Y' or down == 'yes' or down == 'Yes' or down == 'YES' or down == 'True' or down == 'true' or down == 'TRUE':
            down = True
    else:
            down = False
else:
        down = False




# Determine the file matching type:
        
# if downtype matches any of these, only retreive matched raw files:
files1 = ['file', 'files', 'file_only', 'files_only']

# if downtype matches any of these, match files to raw calibrations:        
files2 = ['raw2raw', 'files_calib', 'file_calib', 'file_cal', 'files_cal']

# if downtype matches any of these, match files to master calibrations:        
files3 = ['raw2master', 'files_master', 'file_master', 'file_mast', 'files_mast', 'files_master_calib', 'file_master_calib', 'file_mast_calib', \
          'file_mast_cal', 'files_master_cal', 'files_mast_cal', 'files_mast_calib']


if i_downtype != None:
    downtype  = sys.argv[i_downtype+1]
else:
    downtype  = 'files'

if downtype.lower() in files1:
    downtype = 'files'

if downtype.lower() in files2:
    downtype = 'raw2raw'

if downtype.lower() in files3:
    downtype = 'raw2master'


if (downtype != 'files' and downtype != 'raw2raw' and downtype != 'raw2master'):
    downtype = 'files'


    

print(' ')
print(' ')
print(this_prog, '[info] will query the ESO archive with the following parameters:') 
print(' ')
if verbose == True:
        print('User/Password: %s/%s' %(user, passwd))
print('Target:        %s'    %(target))
print('RA/DEC:        %s/%s' %(ra, dec))
print('Search box:    %s'    %(box_size))
print('RunId:         %s'    %(runid))
print('Instrument:    %s'    %(inst))
print('Start Date:    %s'    %(start))
print('End Date:      %s'    %(end))
print('File Category: %s'    %(catg))
print('File Type:     %s'    %(dpr_type))
print('File Mode:     %s'    %(dpr_tech))
print('MaxRows:       %s'    %(maxr))

if downtype == 'files':
    print('Download Type: %s'    %('Only searched-for raw files will be downloaded'))

if downtype == 'raw2raw':
          print('Download Type: %s'    %('Searched-for raw files + associated raw calibrations will be downloaded'))

if downtype == 'raw2master':
          print('Download Type: %s'    %('Searched-for raw files + associated master calibrations (if available) will be downloaded'))


print(' ')
print(' ')





# =====================================================================================
#
# Querying the archive . . .
#
# =====================================================================================


query_params = {'tab_object':'on',       \
                'target':target,         \
                'resolver':'simbad',     \
                'tab_target_coord':'on', \
                'ra':ra,                 \
                'dec':dec,               \
                'box':box_size,          \
                'deg_or_hour':'hours',   \
                'format':'SexaHours',    \
                'tab_prog_id':'on',      \
                'prog_id':runid,         \
                'tab_instrument':'on',   \
                'instrument':inst,       \
                'stime':start,           \
                'starttime':'0',         \
                'etime':end,             \
                'endtime':'24',          \
                'tab_dp_cat':'true',     \
                'dp_cat':catg,           \
                'tab_dp_tech':'true',    \
                'dp_tech':dpr_tech,      \
                'tab_dp_type':'true',    \
                'dp_type':dpr_type,      \
                'top':maxr,              \
                'tab_rel_date':'0',      \
                'tab_filter_path':'0',   \
                'tab_exptime':'0',       \
                'wdbo':out_format}



getoutput_request = requests.get(url1, params=query_params)
query_output      = getoutput_request.text 



# this gives the number of files found that match the query criteria.  If zero,
# then we might as well quit now:


Nq1          = query_output.find('A maximum of')
Nq2          = query_output.find('criteria') + 9
query_status = query_output[Nq1:Nq2]
Nquery       = [int(s) for s in query_status.split() if s.isdigit()]

if not Nquery:
    Nquery = 0
else:
    Nquery = Nquery[0]


if Nquery == 0:
        print(this_prog, '[info] No records were found matching the provided criteria.  Exiting!')
        print(' ')
        sys.exit()
else:
        print(this_prog, '[info] A total of %i records were found matching the provided criteria.' %(Nquery))
        print(this_prog, '[info] (a maximum limit of %i records were specified in the query ... more may exist)' %(int(maxr)))
        print(' ')



        

# if Nquery > 0, we continue and remove everything but the actual data lines, and clean
# up some lousy multi-word column header names:

end_phrase = 'A maximum of %i records were found matching the provided criteria - any remaining rows were ignored.' %(Nquery)
clean_output = clean_up(query_output, replace_extras={'#':'', end_phrase:'', '\xef\xbb\xbf':'', 'N/A':'NA'})

clean_output = clean_output.replace('Dataset ID', 'Dataset_ID')
clean_output = clean_output.replace('TPL ID', 'TPL_ID')
clean_output = clean_output.replace('TPL START', 'TPL_START')
clean_output = clean_output.replace('DIMM Seeing at Start', 'DIMM_seeing')



# Finally, we must account for the case in which the selected query involves the file category CALIB;
# the DPR.TYPE can be compound words separated by commas (confusing the csv format!).  For example,
# we can have "BIAS,READNOISE", "FLAT,SKY", "FLAT,LAMP", "STD,ZEROPOINT", "IMAGE,JITTER", "FLAT,DOME,LIFETEST", etc.
# All instances of these will be replaced by "FLAT_SKY", "FLAT_LAMP", etc.
# This needs to be fixed before we can proceed.

regex        = re.compile(r'("[^",]*),([^",]*")')                                  # for the double instances like "FLAT,SKY"
clean_output = regex.sub(r'\1_\2', clean_output)

regex        = re.compile(r'"([a-xA-Z0-9,]+),([a-xA-Z0-9,]+),([a-xA-Z0-9,]+)"')    # for the triple instances like "FLAT,DOME,LIFETEST"
clean_output = regex.sub(r'\1_\2_\3', clean_output)

xxx = [x.strip() for x in clean_output.split(',')]



# From the cleaned query output, retreive a list of files (Dataset ID)
# and, for display, the DPR.CATG and DPR.TYPE from each entry line:


n = 14
all_out = [xxx[i * n:(i + 1) * n] for i in range((len(xxx) + n - 1) // n )]


dpr_catg = []
dpr_type = []
dpr_mode = []
files    = []


for i in range(1, Nquery+1):
    dpr_catg.append(all_out[i][5])
    dpr_type.append(all_out[i][6])
    dpr_mode.append(all_out[i][7])
    files.append(all_out[i][8])

dpr_type = [x.replace('"', '') for x in dpr_type]
dpr_mode = [x.replace('"', '') for x in dpr_mode]

Nfiles   = len(files)


if verbose == True:
        print(this_prog, '[info] the following query has been made to %s' %(url1))
        print(' ')
        print('Query Paramters:\n================\n', query_params)
        print(' ')
        print('Query Output:\n=============\n', clean_output)
        print(' ')
        print('Query Files:\n============\n', files)
        print(' ')



print('%30s %15s %20s %20s' %('FileName', 'Category', 'File_Type', 'File_Mode'))
print('%72s' %(90*'='))
for i in range(Nfiles):
    print('%30s %15s %20s %20s' %(files[i], dpr_catg[i], dpr_type[i], dpr_mode[i]))

print(' ')



    
# ====================================================================================
#
# If we have made it thus far, then the query has found something, and can be
# submitted to the archive.
#
# ====================================================================================


if downtype == 'files':
        print('Only these files will be submitted to the archive')
        print(' ')
if downtype == 'raw2raw':
        print('These files will be submitted to the archive and matched to their associated raw calibrations')
        print(' ')
if downtype == 'raw2master':
        print('These files will be submitted to the archive and matched to their associated processed calibrations')
        print(' ')


                
# The files explicitly submitted to the archive must be prefixed with "SAF+" = "SAF%2B" in order to work:                
                
list_files = []

for i in range(Nfiles):

        list_files.append('SAF%2B' + files[i])
        
submit_files = ','.join(list_files)



# ====================================================================================
#
# Submit the file list (files) as an archive request with either:
# (a) only the selected files
# (b) selected files + associated raw calibrations (if available)
# (c) selected files + associated processed calibrations (if available) 
#
# ====================================================================================


url_sub  = url2 + user + '/submission'

# only get the specific data files and update their headers:
if downtype == 'files':
        postdata = 'requestDescription=script&requestCommand=SELECTIVE_HOTFLY&dataset=%s' %(submit_files)

# get the specific data files + raw calibrations and update their headers:
if downtype == 'raw2raw':
        postdata = 'requestDescription=script&requestCommand=SELECTIVE_HOTFLY&requestCommand=CalSelectorRaw2Raw&dataset=%s' %(submit_files)

# get the specific data files + master calibrations and update their headers:
if downtype == 'raw2master':
        postdata = 'requestDescription=script&requestCommand=SELECTIVE_HOTFLY&requestCommand=CalSelectorRaw2Master&dataset=%s' %(submit_files)



submit_call = 'wget -q -O ' + submit_list + ' ' \
               + '--user-agent=' + user_agent + ' --auth-no-challenge ' \
               + '--post-data=' + '"' + postdata + '" ' \
               + '--header="Accept:text/plain" ' \
               + '--http-user=' + user + ' --http-password=' + passwd + ' ' \
               + url_sub



os.system(submit_call)



if verbose == True:
        print(this_prog, '[info] the following request has been submitted using the file created file list:')
        print(' ')
        print(submit_call)
        print(' ')




# ========================================================
# Check that there is a .netrc file in the User's home
# directory
# ========================================================

netrc_path = os.getenv("HOME") + '/.netrc'
newline    = 'machine dataportal.eso.org login ' + user + ' password ' + passwd + '\n'

is_file = os.path.isfile(netrc_path)                # does file exist?

if is_file == True:
        f    = open(netrc_path, 'a+')
        f.write(newline)
        f.close()
else:
        f    = open(netrc_path, 'w')
        f.write(newline)
        f.close()



# ========================================================
#
# From the submission file get the number of the request
#
# ========================================================


try:
        f      = open(submit_list, 'r')
        reqnum = f.read()
        f.close()
except IOError:
        print(this_prog, '[error] the file %s was not found.  Something is wrong with this archive request.  Exiting!' %(submit_list))
        sys.exit()


        
# ========================================================
#
# Make sure that the request is complete such that the
# download.sh script is complete.  Possible states include:
# ERROR
# SUBMITTED
# COMPLETE
#
# ========================================================


url_state      = url2 + user + '/' + reqnum + '/state'

state_call     = 'wget -q -O ' + state_list + ' ' \
               + '--user-agent=' + user_agent + ' --auth-no-challenge ' \
               + '--http-user=' + user + ' --http-password=' + passwd + ' ' \
               + url_state



status = 'INCOMPLETE'
work   = '|'

while status != 'COMPLETE':

        os.system(state_call)

        with open(state_list) as f:
                status = f.readline()
                f.close()

        work = work + '|'


        if status == 'SUBMITTED':
                print(('  waiting for the archive... \r{0:s}'.format(work)), end=' ')
                sys.stdout.flush()
                sleep(1)

        if status == 'ERROR':
                print(this_prog, '[error] there is an error with this archive request.  Exiting!')
                sys.exit()

                
        # this one is intended to catch bad User/Passwords.  In this case, the state file will simply be blank.

        if status == '':
                print(this_prog, '[error] The state of your submitted archive request is blank!')
                print(this_prog, '[error] This probably indicates that either your user name or password is incorrect.')
                print(this_prog, '[error] Please check this.   Exiting!')
                print(' ')
                sys.exit()

                


print(' ')
print(' ')
print(this_prog, '[info] the archive query is now %s' %(status))
print(' ')
print(' ')


        
# ============================================================
#
# Finally, we download the download script from the archive
# and, if requested, download the data . . .
#
# ============================================================


url_down       = url2 + user + '/' + reqnum + '/script'

download_call  = 'wget -q -O ' + down_list + ' ' \
               + '--user-agent=' + user_agent + ' --auth-no-challenge ' \
               + '--http-user=' + user + ' --http-password=' + passwd + ' ' \
               + url_down


os.system(download_call)

os.system('chmod 777 ' + down_list)


if not os.path.exists(down_dir):
        os.makedirs(down_dir)

os.system('mv ' + down_list + ' ' + down_dir)


if down == True:

        print(this_prog, '[info] downloading archive files in %s/' %(down_dir))
        print(' ')
        os.chdir(down_dir)
        os.system('./' + down_list + ' -u ' + user)
        

else:

        print(this_prog, '[info] archive download file can be found in %s/%s' %(down_dir, down_list))
        print(this_prog, '[info] no actual files will be downloaded')
        print(' ')
        print(' ')


        
# =============================================================
# If requested, create a paper trail of files . . .
# (i)  initial query results
# (ii) list of raw files requested
# =============================================================


if record == True:

        query_save = open(out_list, "w")
        query_save.write(query_output.encode('utf-8'))
        query_save.close()

        raw_save   = open(raw_list, "w")
        raw_save.write("\n".join(files))
        raw_save.close()

        print(this_prog, '[info] results of original query saved as %s' %(out_list))
        print(this_prog, '[info] list of requested raw files saved as %s' %(raw_list))
        print(' ')
        print(' ')

        
        

# Cleaning up:

if verbose != True:

        if down == True:
                os.chdir('..')
        
        try:
                os.system('rm ' + submit_list)
        except IOError:
                print(' ')

        try:
                os.system('rm ' + state_list)
        except IOError:
                print(' ')
