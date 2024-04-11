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


class V20Seq1DietIt350(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 500
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 350

class MipV20Seq1It350(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 4)


        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 500
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 350

class MipV20Seq1It100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 4)


        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 300
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100

class MipV20Seq1It100Begin100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 4)


        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 100
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100

class MipV10Seq1It100Begin100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 2)


        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 10 - base

        it_base = base * 100
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100

class MipFixedV20(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 20)

        self.load_its = {}

class MipFixedN(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, N, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), N)

        self.load_its = {}


class MipV30Seq1It100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        dataset_size = kwargs.get("dataset_size")
        # Sample with a fixed random seed and without replacement
        self.init_views = random.Random(0).sample(list(range(dataset_size)), 4)


        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 300
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100


class All(BaseSchema):

    def __init__(self, **kwargs) -> None:
        dataset_size = kwargs.get("dataset_size")
        self.init_views = list(range(dataset_size))
        random.shuffle(self.init_views)
        self.load_its = {}
    
    def num_views_to_add(self, it: int) -> int:
        return 0

class Skip10(BaseSchema):

    def __init__(self, **kwargs) -> None:
        dataset_size = kwargs.get("dataset_size")
        self.init_views = [idx for idx in range(dataset_size) if idx % 10 ==0]
        random.shuffle(self.init_views)
        self.load_its = {}
    
    def num_views_to_add(self, it: int) -> int:
        return 0

class Fixed100(BaseSchema):

    def __init__(self, **kwargs) -> None:
        self.init_views = list(range(100))
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
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 1 
        for i in range(num_views_left):
            self.load_its[it_base] = 4

            base += 1
            it_base += base * 1

class V20Seq1DietIt100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 500
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100


class V20Seq1FVSIt100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [0, 25, 50, 99] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 100
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100

class VNSeqMInplaceFVS(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, dataset_size: int, scene, N: int=20, M: int=1, num_init_views: int=4, interval_epochs=100, **kwargs):
        """
        N: int total views to select
        M: # views to select each time
        """
        super().__init__()
        self.init_views = [0] # furtherest views found by algorithms #[0, 25, 50, 99]

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

V20Seq1InplaceFVS = partial(VNSeqMInplaceFVS, N=20, M=1, num_init_views=4)
V10Seq1InplaceFVS = partial(VNSeqMInplaceFVS, N=10, M=1, num_init_views=2)

V20Seq4InplaceFVS = partial(VNSeqMInplaceFVS, N=20, M=4, num_init_views=4, interval_epochs=300)
V10Seq2InplaceFVS = partial(VNSeqMInplaceFVS, N=10, M=2, num_init_views=2, interval_epochs=300)

FixedV20FVS = partial(VNSeqMInplaceFVS, N=20, M=0, num_init_views=20)
V32Seq4InplaceFVS = partial(VNSeqMInplaceFVS, N=32, M=4, num_init_views=4, interval_epochs=300)


class VNSeqMSection(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, dataset_size: int, scene, N: int=20, M: int=1, num_init_views: int=4, num_chunks: int=10, interval_epochs=100, **kwargs):
        """
        N: int total views to select
        M: # views to select each time
        """
        super().__init__()
        self.load_its = {}

        self.candidate_views_filter = {}
        
        # NOTE: following the default implementation of scene. 
        # result of getTrainCameras are shuffled in-place, thus we have to rely on train_idxs to index the views
        all_cams = scene.train_cameras[1.0]
        assert N == num_chunks * M, "Other combination not implemented yet"

        # find num_chunks furtherest points
        anchor_idxs = [0]
        candidate_idxs = [i for i in range(len(all_cams)) if i not in anchor_idxs]
        anchor_positions = [all_cams[i].camera_center.cpu() for i in anchor_idxs]
        candidate_positions = [all_cams[i].camera_center.cpu() for i in candidate_idxs]
        # NOTE: we use cpu pytorch to make sure it's determinstic
        # although torch.use_deterministic_algorithms(True) shows argmax can also be deterministic on CUDA

        while len(anchor_idxs) < num_chunks:
            dist_mat = torch.cdist(torch.stack(candidate_positions), torch.stack(anchor_positions))
            candidate_dist = reduce(dist_mat, "c t -> c", "min")

            selected_idx = candidate_dist.argmax().item()
            anchor_idxs.append(candidate_idxs.pop(selected_idx))
            anchor_positions.append(candidate_positions.pop(selected_idx))

        chunked_cam_idxs = [[i] for i in anchor_idxs]

        # put remaining cams into each cloest chunk
        dist_mat = torch.cdist(torch.stack(candidate_positions), torch.stack(anchor_positions))

        candidate_chunk_idxs = dist_mat.argmin(1)

        for rel_idx, matched_chunk_idx in enumerate(candidate_chunk_idxs):
            chunked_cam_idxs[matched_chunk_idx].append(candidate_idxs[rel_idx])

        assert all(M <= len(chunk) for chunk in chunked_cam_idxs), "chunk size should be larger than M"
        
        chunked_init_views = [random.Random(0).sample(chunk, M) for chunk in chunked_cam_idxs[:num_init_views]]
        self.init_views = list(itertools.chain(*chunked_init_views))

        cur_dataset_size = len(self.init_views)
        it_base = cur_dataset_size * interval_epochs
        num_views_left = N - len(self.init_views)
        
        chunk_id = len(self.init_views) // M

        if num_views_left > 0:
            assert num_views_left % M == 0, "cannot split M evenly to the rest views"

            while num_views_left > 0:
                self.load_its[it_base] = M

                assert chunk_id < len(chunked_cam_idxs)
                # need clouser because lambda doesn't work
                def filter_idxs(chunk):
                    return lambda x: x in chunk
                self.candidate_views_filter[it_base] = filter_idxs(chunked_cam_idxs[chunk_id])
                # self.candidate_views_filter[it_base] = lambda x: x in chunked_cam_idxs[chunk_id]

                chunk_id = chunk_id + 1
                cur_dataset_size += M
                it_base += cur_dataset_size * interval_epochs
                num_views_left -= M

V10Seq1Section10 = partial(VNSeqMSection, N=10, M=1, num_init_views=2, num_chunks=10)
V20Seq1Section20 = partial(VNSeqMSection, N=20, M=1, num_init_views=4, num_chunks=20)
V20Seq2Section10 = partial(VNSeqMSection, N=20, M=2, num_init_views=4, num_chunks=10)


class V10Seq1DietIt100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [26, 86] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 20 - base

        it_base = base * 300
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100

class V15Seq1DietIt100(BaseSchema):
    """
    Add 1 image at a time
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_views = [26, 86] # furtherest views found by algorithms #[0, 25, 50, 99]

        self.load_its = {}
        base = len(self.init_views)
        num_views_left = 15 - base

        it_base = base * 300
        for i in range(num_views_left):
            self.load_its[it_base] = 1

            base += 1
            it_base += base * 100


schema_dict: Dict[str, BaseSchema] = {'v20seq1diet_it350': V20Seq1DietIt350, 'all': All, "debug": V20Seq1Debug,
                                      'v20seq1diet_it100': V20Seq1DietIt100, 'v10seq1diet_it100': V10Seq1DietIt100, 
                                      'v15seq1diet_it100': V15Seq1DietIt100, 
                                      'mipv20seq1_it350': MipV20Seq1It350, 'mipv20seq1_it100': MipV20Seq1It100,
                                      'mipv30seq1_it100': MipV30Seq1It100, "mipv20seq1_it100begin100": MipV20Seq1It100Begin100,
                                      "mipv20fixed": MipFixedV20, "v20seq1fvs_it100": V20Seq1FVSIt100,
                                      "mipv10seq1_it100": MipV10Seq1It100Begin100,
                                      "v20seq1_inplace": V20Seq1InplaceFVS, "v10seq1_inplace": V10Seq1InplaceFVS,
                                      "fixed20fvs": FixedV20FVS, "v20seq4_inplace": V20Seq4InplaceFVS,
                                      "v10seq2_inplace": V10Seq2InplaceFVS, 
                                      "v10seq1section10": V10Seq1Section10,  "v20seq1section20": V20Seq1Section20,
                                      "v20seq2section10": V20Seq2Section10, "v32seq4_inplace": V32Seq4InplaceFVS,
                                      "skip10": Skip10,
                                      }

extra_schemas: Dict[str, BaseSchema] = {}

for views in [10, 20, 30, 40, 50,]:
    extra_schemas[f"mipv{views}fixed"] = partial(MipFixedN, N=views)

schema_dict.update(extra_schemas)


override_test_idxs_dict: Dict[str, List[int]] = {"basket": list(range(42, 50,2)), "africa": list(range(6, 14, 2)),
                                            "statue": list(range(68, 76, 2)), "torch": list(range(9, 17, 2))}

override_train_idxs_dict: Dict[str, List[int]] = {"basket": list(range(43, 50,2)), "africa": list(range(5, 14, 2)),
                                            "statue": list(range(67, 76, 2)), "torch": list(range(8, 17, 2))}


class ActiveNeRF20():
    def __init__(self, ):
        self.init_views = [42, 89, 85, 95] # furtherest views found by algorithms #[0, 25, 50, 99]
        # self.load_epoch = {200: 4 , 400: 4, 800: 4, 1000: 4}
        self.load_epoch = {1000: 4 , 2000: 4, 3000: 4, 4000: 4}
        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0

class ActiveNeRF20GH():
    def __init__(self, ):
        self.init_views = [0, 5, 10, 15] 
        self.load_epoch = {1000: 4 , 2000: 4, 3000: 4, 4000: 4}

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0

class ActiveNeRF20GH_debug():
    def __init__(self, ):
        self.init_views = [0, 5, 10, 15] 
        self.load_epoch = {30: 4 , 35: 4, 40: 4, 50: 4}

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0


class ActiveNeRF10():
    def __init__(self, ):
        self.init_views = [64, 65] # furtherest views found by algorithms #[0, 25, 50, 99]
        self.load_epoch = {1000: 2 , 2000:2, 3000:2, 4000:2}
        # self.load_epoch = {20: 2 , 400:2, 800:2, 1000:2}

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0


class V10Seq1():
    def __init__(self, ):
        self.init_views = [64, 65] # furtherest views found by algorithms #[0, 25, 50, 99]
        # self.load_epoch = {500: 1, 750: 1, 1000: 1, 1250: 1, 1500: 1, 1750: 1, 2000: 1, 2250: 1}
        self.load_epoch = {500: 1, 1000: 1, 1500: 1, 2000: 1, 2500: 1, 3000: 1, 3500: 1, 4000: 1}
        # self.load_epoch = {20: 2 , 400:2, 800:2, 1000:2}

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0


class V20Seq1Diet():
    """
    Add 1 image at a time
    """

    def __init__(self, ):
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        # For debugging
        self.load_epoch = {1000: 1, 1250:1, 1500:1, 1750:1, 2000: 1, 2250: 1, 2500:1, 2750:1,
        # self.load_epoch = {10: 1, 1250:1, 1500:1, 1750:1, 2000: 1, 2250: 1, 2500:1, 2750:1,
                            3000: 1, 3250: 1, 3500: 1, 3750:1, 
                            4000: 1, 4250: 1, 4500: 1, 4750:1}
        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0

class V20Seq1DietEp10():
    """
    Add 1 image at a time
    """

    def __init__(self, ):
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        # For debugging
        self.load_epoch = {10: 1, 20:1, 30:1, 40:1, 50: 1, 60: 1, 70:1, 80:1,
                            90: 1, 100: 1, 110: 1, 120:1, 
                            130: 1, 140: 1, 150: 1, 160:1}
        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0


class V20Seq1DietEp150():
    """
    Add 1 image at a time
    """

    def __init__(self, ):
        self.init_views = [26, 86, 2, 55] # furtherest views found by algorithms #[0, 25, 50, 99]

        # For debugging
        self.load_epoch = {(i + 1) * 150: 1 for i in range(20 - len(self.init_views))}

        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0


class V20Seq1():
    """
    Add 1 image at a time
    """

    def __init__(self, ):
        self.init_views = [42, 89, 85, 95] # furtherest views found by algorithms #[0, 25, 50, 99]

        # For debugging
        self.load_epoch = {1000: 1, 1250:1, 1500:1, 1750:1, 2000: 1, 2250: 1, 2500:1, 2750:1,
        # self.load_epoch = {10: 1, 1250:1, 1500:1, 1750:1, 2000: 1, 2250: 1, 2500:1, 2750:1,
                            3000: 1, 3250: 1, 3500: 1, 3750:1, 
                            4000: 1, 4250: 1, 4500: 1, 4750:1}
        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0

class V20Seq1Long():
    """
    Add 1 image at a time
    Not used yet
    """

    def __init__(self, ):
        self.init_views = [42, 89, 85, 95] # furtherest views found by algorithms #[0, 25, 50, 99]

        # For debugging
        self.load_epoch = {1000: 1, 1250:1, 1500:1, 1750:1, 2000: 1, 2250: 1, 2500:1, 2750:1,
                            3000: 1, 3250: 1, 3500: 1, 3750:1, 
                            4000: 1, 4250: 1, 4500: 1, 4750:1}
        # self.load_epoch = {20: 4 , 400:4, 800:4, 1000:4} # Fox debugging

    
    def num_views_to_add(self, epoch) -> int:
        if epoch in self.load_epoch:
            return self.load_epoch[epoch]
        else:
            return 0



