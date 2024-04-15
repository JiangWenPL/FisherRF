import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import torch
import argparse
import os
from glob import glob
import seaborn as sns

# Example
# python ause.py gaussian africa 68 0
# python ause.py [method] [object] [view id] [output id]

def equal_hist(uncern):
    H, W = uncern.shape

    # Histogram equalization for visualization
    uncern = uncern.flatten()
    median = np.median(uncern)
    bins = np.append(np.linspace(uncern.min(), median, len(uncern)), 
                            np.linspace(median, uncern.max(), len(uncern)))
    # Do histogram equalization on uncern  
    # bins = np.linspace(uncern.min(), uncern.max(), len(uncern) // 20)
    hist, bins2 = np.histogram(uncern, bins=bins)
    # Compute CDF from histogram
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf = np.hstack(([0], cdf))
    cdf = cdf / cdf[-1]
    # Do equalization
    binnum = np.digitize(uncern, bins, True) - 1
    neg = np.where(binnum < 0)
    binnum[neg] = 0
    uncern_aeq = cdf[binnum] * bins[-1]

    uncern_aeq = uncern_aeq.reshape(H, W)
    uncern_aeq = (uncern_aeq - uncern_aeq.min()) / (uncern_aeq.max() - uncern_aeq.min())
    return uncern_aeq 

def tensor_erode(bin_img, ksize=5):
    import pdb; pdb.set_trace()
    H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    # unfold into patch
    patches = bin_img.unfold(dimension=0, size=ksize, step=1)
    patches = patches.unfold(dimension=1, size=ksize, step=1)
    # B x C x H x W x k x k

    # take min value
    eroded, _ = patches.reshape(H, W, -1).min(dim=-1)
    return eroded

arg = argparse.ArgumentParser()
arg.add_argument("method", type=str)
arg.add_argument("obj", type=str)
arg.add_argument("--idx", type=str, default=None)
arg.add_argument("--data_dir", type=str, default="/mnt/kostas-graid/datasets/wen/LF")
arg.add_argument("-m", "--model_path", type=str, default=None)
arg.add_argument("--viz", action="store_true")
arg.add_argument("--auto_scale", action="store_true")
arg.add_argument("--foreground", action="store_true")
opt = arg.parse_args()

for index in range(4):
    obj = opt.obj
    depth_gt_file = os.path.join(opt.data_dir, "{}/depth_gt_{:02d}.npy".format(opt.obj, index))
    depth_gt = np.ascontiguousarray(np.load(depth_gt_file))
    
    if opt.method == "cfnerf":
        depth_pred_file = "./cfnerf_{}/depth_{:03d}.npz".format(obj, index)
        uncertainty_file =  "./cfnerf_{}/uncern_{:03d}.npz".format(obj, index)
        print(depth_pred_file, uncertainty_file)
        if obj == "statue":
            scale = 1.1
        elif obj == "africa":
            scale = 4.9
        elif obj == "torch":
            scale = 6.
        else:
            scale = 1
        
        depth_pred = np.load(depth_pred_file)["pred"]

        if opt.auto_scale:
            scale = (np.median(depth_gt) / np.median(depth_pred))
            print(f"scale: {scale}")
        
        depth_pred = depth_pred * scale
        uncern = np.load(uncertainty_file)["pred"]
        uncern = np.log(uncern)


    if opt.method == "gaussian":
        if opt.model_path is None:
            pred_files = sorted(glob(f"./gaussian_{obj}/uncertainty_*.npz"))
        else:
            pred_files = sorted(glob(f"{opt.model_path}/eval/*.npz"))

        pred_file = pred_files[index]
        data = np.load(pred_file)
        
        if obj == "statue":
            scale = 1.25   
        elif obj == "torch":
            scale = 10
        elif obj == "africa":
            scale = 4
        elif obj == "basket":
            scale = 1 / 8
        else:
            scale = 1
        
        uncern = torch.from_numpy(data["uncertanity_map"])
        pixel_gaussian_counter = data["pixel_gaussian_counter"]
        uncern = F.interpolate(uncern[None, None, ...], depth_gt.shape, mode="nearest")[0, 0].numpy()
        uncern = np.log(uncern)
        uncern = np.where(np.isinf(uncern), uncern.max(), uncern)
        
        depth_pred = torch.from_numpy(data["depth"])
        depth_pred = F.interpolate(depth_pred[None, None, ...], depth_gt.shape, mode="nearest")[0, 0] 

        if opt.auto_scale:
            scale = (np.median(depth_gt) / np.median(depth_pred))
            print(f"scale: {scale}")

        depth_pred = depth_pred.numpy() * scale

    if opt.method == 'plenoxel':
        assert opt.foreground, " plenoxel can only be tested using foreground. "
        depth_file = sorted(glob(f"plenoxel_{opt.obj}/*_depth.npz"))[index]
        uncern_file = sorted(glob(f"plenoxel_{opt.obj}/*_uncern.npz"))[index]

        depth_pred = np.load(depth_file)["pred"][::2, ::2]
        uncern = np.load(uncern_file)["pred"][::2, ::2]

        if obj == "statue":
            scale = 1   
        elif obj == "torch":
            scale = 6
        elif obj == "africa":
            scale = 4
        elif obj == "basket":
            scale = 18
        else:
            scale = 1

        depth_pred = depth_pred * scale

    if opt.method == 'ActiveNeRF':
        depth_file = sorted(glob(f"ActiveNeRF_{opt.obj}_4/*_depth.npz"))[index]
        uncern_file = sorted(glob(f"ActiveNeRF_{opt.obj}_4/*_uncert.npz"))[index]

        depth_pred = np.load(depth_file)["depth"]
        depth_pred = np.where(np.isnan(depth_pred), 100., depth_pred)
        uncern = np.load(uncern_file)["uncern"]
        H, W = uncern.shape

        # Do histogram equalization on uncern  
        uncern = uncern.flatten()
        bins = np.linspace(uncern.min(), uncern.max(), len(uncern) // 20)
        hist, bins2 = np.histogram(uncern, bins=bins)
        # Compute CDF from histogram
        cdf = np.cumsum(hist, dtype=np.float64)
        cdf = np.hstack(([0], cdf))
        cdf = cdf / cdf[-1]
        # Do equalization
        binnum = np.digitize(uncern, bins, True) - 1
        neg = np.where(binnum < 0)
        binnum[neg] = 0
        uncern_aeq = cdf[binnum] * bins[-1]
        
        uncern = uncern_aeq.reshape(H, W)

        if obj == "statue":
            scale = 1.
        elif obj == "africa":
            scale = 4.9
        elif obj == "torch":
            scale = 6.
        elif obj == "basket":
            scale = 7.2
        else:
            scale = 1

        depth_pred = depth_pred * scale

    depth_gt, depth_pred, uncern = torch.from_numpy(depth_gt), torch.from_numpy(depth_pred), torch.from_numpy(uncern)

    depth_error_map = torch.abs(depth_pred - depth_gt)
    print("depth MAE: ", depth_error_map.mean().item())

    if opt.viz:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        im1 = ax1.imshow(depth_error_map)
        im2 = ax2.imshow(depth_error_map)
        ax3.imshow(depth_pred)
        ax4.imshow(depth_gt)

        ax5.imshow(depth_error_map)
        ax6.imshow((uncern))

        # Initialization function: plot the background of each frame
        def init():
            return im1, im2
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(depth_pred, cmap="crest", square=True, cbar=False)
    plt.axis('off')
    plt.savefig("{}_{}_{}_depth_pred.png".format(opt.method, opt.obj, index))

    plt.figure(figsize=(16, 10))
    sns.heatmap(equal_hist(depth_error_map), square=True, cbar=False)
    plt.axis('off')
    plt.savefig("{}_{}_{}_depth_error.png".format(opt.method, opt.obj, index))

    plt.figure(figsize=(16, 10))
    sns.heatmap(equal_hist(uncern), square=True, cbar=False)
    plt.axis('off')
    plt.savefig("{}_{}_{}_uncern.png".format(opt.method, opt.obj, index))

    err_vec = depth_error_map.reshape(-1)
    unc_vec = uncern.reshape(-1)

    # Sort the error
    ratio_removed = torch.linspace(0, 0.999, 100)
    
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        err = err_slice.mean().numpy()
        if np.isnan(err):
            continue
        else:
            ause_err.append(err)

    
    unc_vec_sorted, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        err = err_slice.mean().numpy()
        if np.isnan(err):
            continue
        else:
            ause_err_by_var.append(err)

    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)

    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed[:len(ause_err)])
    print(f"ause: {ause}")

    if opt.viz:
        def update(frame):
            r = ratio_removed[frame]
            idx = min(int((1-r)*n_valid_pixels), len(err_vec_sorted) - 1)
            err_dthresh = err_vec_sorted[idx]
            err_uthresh = unc_vec_sorted[idx]
            
            mask_dthresh = depth_error_map <= err_dthresh
            mask_uthresh = uncern <= err_uthresh

            im1.set_data(mask_dthresh * depth_error_map)
            im2.set_data(mask_uthresh * depth_error_map)

        # Call the animator. blit=True means only re-draw the parts that have changed.
        ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

        fig.show()
        input("")
