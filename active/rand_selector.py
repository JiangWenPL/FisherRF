import torch
import numpy as np
from typing import List, Dict, Union, Optional
from copy import deepcopy
import random

class RandSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed

    
    def nbvs(self, gaussian, scene, num_views, *args, **kwargs) -> List[int]:
        candidate_views = deepcopy(list(scene.get_candidate_set()))
        random.Random(self.seed).shuffle(candidate_views)

        return candidate_views[:num_views]
    
    def forward(self, x):
        return x