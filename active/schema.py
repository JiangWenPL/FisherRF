# Regulator for active learning policies
from typing import List, Dict
import random
from functools import partial
import torch
from copy import deepcopy
from einops import rearrange, reduce, repeat
import math
import itertools

class BaseSchema:
    init_views: List[int]
    load_its: Dict[int, int]

    def __init__(self, **kwargs) -> None:
        self.init_views = []
        self.load_its = {}


    def num_views_to_add(self, it:int) -> int:
        return self.load_its.get(it, 0)

class All(BaseSchema):

    def __init__(self, **kwargs) -> None:
        dataset_size = kwargs.get("dataset_size")
        self.init_views = list(range(dataset_size))
        random.shuffle(self.init_views)
        self.load_its = {}
    
    def num_views_to_add(self, it: int) -> int:
        return 0


class V20Seq1Debug(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [26, 86, 2, 55]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 1 
        for i in range(num_views_left):
            self.load_its[it_base] = 4

            base += 1
            it_base += base * 1

class VNSeqMInplace(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, dataset_size: int, scene, N: int=20, M: int=1, num_init_views: int=4, interval_epochs=100, **kwargs):
        """
        N: int total views to select
        M: # views to select each time
        """
        super().__init__()
        self.init_views = [0] 

        self.load_its = {}
        num_init_views_needed = num_init_views - len(self.init_views)
        
        # NOTE: following the default implementation of scene. 
        # result of getTrainCameras are shuffled in-place, thus we have to rely on train_idxs to index the views
        all_cams = scene.train_cameras[1.0]
        candidate_views = [i for i in range(len(all_cams)) if i not in self.init_views]
        train_cams = [all_cams[i] for i in self.init_views]
        candidate_cams = [all_cams[i] for i in candidate_views]

        selected_idxs = []

        for _ in range(num_init_views_needed):
            trainT = torch.stack([i.camera_center.cpu() for i in train_cams])
            candidateT = torch.stack([i.camera_center.cpu() for i in candidate_cams])

            dist_mat = torch.cdist(candidateT, trainT)
            candidate_dist = reduce(dist_mat, "c t -> c", "min")

            selected_idx = candidate_dist.argmax().item()
            selected_idxs.append(candidate_views.pop(selected_idx))

            # Put selected cam into training cam
            train_cams.append(candidate_cams.pop(selected_idx))
        
        self.init_views.extend(selected_idxs)

        cur_dataset_size = len(self.init_views)
        it_base = cur_dataset_size * interval_epochs
        num_views_left = N - len(self.init_views)
        
        if num_views_left > 0:
            assert num_views_left % M == 0, "cannot split M evenly to the rest views"

            while num_views_left > 0:
                self.load_its[it_base] = M

                cur_dataset_size += M
                it_base += cur_dataset_size * interval_epochs
                num_views_left -= M

V20Seq1Inplace = partial(VNSeqMInplace, N=20, M=1, num_init_views=4)
V10Seq1Inplace = partial(VNSeqMInplace, N=10, M=1, num_init_views=2)
V20Seq4Inplace = partial(VNSeqMInplace, N=20, M=4, num_init_views=4, interval_epochs=300)




schema_dict: Dict[str, BaseSchema] = {'all': All, "debug": V20Seq1Debug,
                                      "v20seq1_inplace": V20Seq1Inplace, "v10seq1_inplace": V10Seq1Inplace,
                                      "v20seq4_inplace": V20Seq4Inplace,
                                      }

override_test_idxs_dict: Dict[str, List[int]] = {"basket": list(range(42, 50,2)), "africa": list(range(6, 14, 2)),
                                            "statue": list(range(68, 76, 2)), "torch": list(range(9, 17, 2))}

override_train_idxs_dict: Dict[str, List[int]] = {"basket": list(range(43, 50,2)), "africa": list(range(5, 14, 2)),
                                            "statue": list(range(67, 76, 2)), "torch": list(range(8, 17, 2))}