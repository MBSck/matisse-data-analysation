__author__ = "Marten Scheuck"

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.io import ascii
from astropy import units as u 

# Queries the simbad catalog for an object
result_table = Simbad.query_object("HD39853")
print(result_table)

mdfc = Vizier(catalog="II/361/mdfc-v10",columns=["**"])
result_mdfc = mdfc.query_object("HD39853", catalog=["II/361/mdfc-v10"], radius=5.0*u.arcsec)
print(result_mdfc[0]["med-Lflux"])
print(result_mdfc[0]["med-Nflux"])

