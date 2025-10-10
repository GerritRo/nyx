from nyx.core import get_parameter
from .airmass import plane_parallel, kasten_young_1989

AIRMASS_FUNCTIONS = {
    "plane_parallel": plane_parallel,
    "kasten_young_1989": kasten_young_1989
}

def get_airmass_formula():
    conf_par = get_parameter('airmass_formula')
    return AIRMASS_FUNCTIONS[conf_par]

