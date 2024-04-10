from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .V_sel import VarSelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "variance": VarSelector}