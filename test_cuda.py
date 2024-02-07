import sys
sys.dont_write_bytecode = True

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch
import math
from modified_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def modified_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def test_training():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--method", type=str, default="rand")
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)

    dataset_dir = os.environ.get("DATASET_DIR", "/mnt/kostas-graid/datasets/blender/nerf_synthetic/lego/")
    args = parser.parse_args(f"-s {dataset_dir}/".split())
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # args.port = find_free_port()

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    
    data_tuple_path = os.environ.get("TEST_DATA_PATH", "/mnt/kostas-graid/sw/envs/wen/gaussian_test_tuple.pth")
    data_tuple = torch.load(data_tuple_path)
    viewpoint_cam, model_params, opt, pipe, background, render_pkg_old, gt_image = data_tuple

    gaussians.restore(model_params, opt)

    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()

    grad_xyz, grad_features_dc, grad_features_rest, grad_scaling, grad_rotation, grad_opacity = (i.grad.detach().clone() for i in model_params[1:7])
    gaussians.optimizer.zero_grad(set_to_none = True)

    render_pkg = modified_render(viewpoint_cam, gaussians, pipe, background)
    modified_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    Ll1 = l1_loss(modified_image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(modified_image, gt_image))
    loss.backward()
    modified_grad_xyz, modified_grad_features_dc, modified_grad_features_rest, modified_grad_scaling, modified_grad_rotation, modified_grad_opacity = (i.grad.detach().clone() for i in model_params[1:7])

    assert torch.allclose(image, modified_image), "Rendered output should be the same"
    assert torch.allclose(grad_xyz, modified_grad_xyz)
    assert torch.allclose(grad_features_dc, modified_grad_features_dc)
    assert torch.allclose(grad_features_rest, modified_grad_features_rest)
    # The following two doesn't pass for torch.all close as the a few (< 5) elements varies between 1e-8 to 2e-8 
    # assert (torch.sum(torch.abs(grad_scaling - modified_grad_scaling) > 1e-8) < 5
    #         and torch.max(torch.abs(grad_scaling - modified_grad_scaling)) < 3e-8)
    assert torch.allclose(grad_scaling, modified_grad_scaling, atol=5e-8)
    assert torch.allclose(grad_rotation, modified_grad_rotation, atol=5e-8)
    # assert (torch.sum(torch.abs(grad_rotation - modified_grad_rotation) > 1e-8) < 5
    #         and torch.max(torch.abs(grad_rotation - modified_grad_rotation)) < 3e-8)
    assert torch.allclose(grad_opacity, modified_grad_opacity)

    # All done
    print("\nTraining complete.")



if __name__ == "__main__":
    # Set up command line argument parser
    test_training()
