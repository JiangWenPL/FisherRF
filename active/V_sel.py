import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render_variance, network_gui
from scene import Scene
import math
import os

class VarSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        # off load to cpu to avoid oom with greedy algo
        # device = params[0].device if num_views == 1 else "cpu"
        device = "cpu" # we have to load to cpu because of inflation
        candidate_cameras = scene.getCandidateCameras()

        acq_scores = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating Variance on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = render_variance(cam, gaussians, pipe, background)
            acq_score = (render_pkg["pri_var"] - render_pkg["post_var"]).sum()
            acq_scores.append(acq_score.item())
        
        acq_scores = np.array(acq_scores)
        selected_idxs = np.argsort(acq_scores)[-num_views:]
        selected_view_idx = [candidate_views[k] for k in selected_idxs]
        return selected_view_idx
    
    def forward(self, x):
        return x