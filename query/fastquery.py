#!/usr/bin/env python3

__author__ = "Marten Scheuck"

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy import units as u
from astropy.coordinates import SkyCoord

# Returned fields:
# SIMBAD: 'MAIN_ID', 'RA', 'DEC', 'RA_PREC', 'DEC_PREC', 'COO_ERR_MAJA',
# 'COO_ERR_MINA', 'COO_ERR_ANGLE', 'COO_QUAL', 'COO_WAVELENGTH', 'COO_BIBCODE',
# 'MK_ds', 'MK_mss', 'MK_Spectral_type', 'MK_bibcode', 'SP_TYPE', 'SP_TYPE_2',
# 'SP_QUAL', 'SP_BIBCODE', 'Fe_H_Teff', 'Fe_H_log_g', 'Fe_H_Fe_H', 'Fe_H_flag',
# 'Fe_H_CompStar', 'Fe_H_CatNo', 'Fe_H_bibcode', 'PMRA', 'PMDEC', 'PM_ERR_MAJA',
# 'PM_ERR_MINA', 'PM_ERR_ANGLE', 'PLX_VALUE', 'RV_VALUE', 'FLUX_U', 'FLUX_ERROR_U',
# 'FLUX_B', 'FLUX_ERROR_B', 'FLUX_V', 'FLUX_ERROR_V', 'FLUX_R', 'FLUX_ERROR_R',
# 'FLUX_I', 'FLUX_ERROR_I', 'FLUX_J', 'FLUX_ERROR_J', 'FLUX_H', 'FLUX_ERROR_H',
# 'FLUX_K', 'FLUX_ERROR_K'

# Gaia: 'RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Source', 'Plx', 'e_Plx',
#   'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'Dup', 'FG', 'e_FG', 'Gmag', 'e_Gmag',
#   'FBP', 'e_FBP', 'BPmag', 'e_BPmag', 'FRP', 'e_FRP', 'RPmag', 'e_RPmag', 'BP-RP',
#    'RV', 'e_RV', 'Teff', 'AG', 'E_BP-RP_', 'Rad', 'Lum'

# Tycho: 'e_BTmag', 'e_VTmag', 'IDTyc2', 'RAT', 'DET', 'pmRA', 'pmDE', 'EpRA1990',
#   'EpDE1990', 'CCDM', 'BTmag', 'VTmag', 'TDSC', 'WDS', 'PA', 'Sep', '_RA.icrs', '_DE.icrs'

# NOMAD: 'NOMAD1', 'YM', 'RAJ2000', 'DEJ2000', 'r', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE',
#   'Bmag', 'r_Bmag', 'Vmag', 'r_Vmag', 'Rmag', 'r_Rmag', 'Jmag', 'Hmag', 'Kmag', 'R'

# 2MASS: 'RAJ2000', 'DEJ2000', '_2MASS', 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag',
#   'Qflg', 'Rflg', 'Bflg', 'Cflg', 'Xflg', 'Aflg'

# WISE: 'RAJ2000','DEJ2000','eeMaj','eeMin','Im','W1mag','e_W1mag','W2mag','e_W2mag',
#   'W3mag','e_W3mag','W4mag','e_W4mag','Jmag','Hmag','Kmag','ccf','ex','var','d2M','_2M'

# MDFC: '_r', 'Name', 'SpType', 'RAJ2000', 'DEJ2000', 'Dist', 'Teff-MIDI', 'Teff-GAIA',
#   'Comp', 'Mean-sep', 'mag1', 'mag2', 'Diam-MIDI', 'e_Diam-MIDI', 'Diam-Cohen',
#   'e_Diam-Cohen', 'Diam-GAIA', 'LDD-meas', 'e_LDD-meas', 'UDD-meas', 'Band-meas',
#   'LDD-est', 'e_LDD-est', 'UDDL-est', 'UDDM-est', 'UDDN-est', 'Jmag', 'Hmag', 'Kmag',
#   'W4mag', 'CalFlag', 'IRflag', 'nb-Lflux', 'med-Lflux', 'disp-Lflux', 'nb-Mflux',
#   'med-Mflux', 'disp-Mflux', 'nb-Nflux', 'med-Nflux', 'disp-Nflux', 'Lcorflux30',
#   'Lcorflux100', 'Lcorflux130', 'Mcorflux30', 'Mcorflux100', 'Mcorflux130', 'Ncorflux30',
#   'Ncorflux100', 'Ncorflux130', 'Simbad'

catalogs = {"simbad": Simbad(), "gaia": Vizier(catalog="I/345/gaia2"), "tycho": Vizier(catalog="I/350/tyc2tdsc",columns = ["*","e_BTmag","e_VTmag"]),
             "nomad": Vizier(catalog="I/297/out"), "twomass": Vizier(catalog="II/246/out"), "wise": Vizier(catalog="II/311/wise"),
            "mdfc": Vizier(catalog="II/361/mdfc-v10",columns=["**"])}

def query(obj_name: str, header_name: str, catalog: str, match_radius: float = 5.):
    """Queries the simbad catalog"""
    if catalog == "simbad":
        catalogs["simbad"].add_votable_fields('mk','sp','sptype','fe_h','pm','plx','rv_value',
            'flux(U)','flux_error(U)',
            'flux(B)','flux_error(B)',
            'flux(V)','flux_error(V)',
            'flux(R)','flux_error(R)',
            'flux(I)','flux_error(I)',
            'flux(J)','flux_error(J)',
            'flux(H)','flux_error(H)',
            'flux(K)','flux_error(K)')

        result = catalogs["simbad"].query_object(obj_name)

    if catalog == "mdfc":
        result = catalogs["mdfc"].query_object(obj_name,catalog=["II/361/mdfc-v10"],radius=match_radius*u.arcsec)

    if catalog == "nomad":
        result = catalogs["nomad"].query_object(obj_name,catalog=["I/297/out"],radius=match_radius*u.arcsec)

    if catalog == "wise":
        result = catalogs["wise"].query_object(obj_name,catalog=["II/311/wise"],radius=match_radius*u.arcsec)

    return result[0][header_name]

if __name__ == "__main__":
    star = "HD_100546"

    # TODO: Add function that automatically gets that information, checks for
    # errors if some value cannot be gotten and tries the other databases
    print(query(star, "med-Lflux", "mdfc"))

