#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, modified_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from active.schema import schema_dict, override_test_idxs_dict, override_train_idxs_dict
from utils.loss_utils import ssim
from lpipsPyTorch import lpips, lpips_func
from active import methods_dict
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.cluster_manager import ClusterStateManager

csm = ClusterStateManager()

@torch.no_grad()
def save_checkpoint(gaussians, iteration, scene, base_iter=0, save_path=None, save_last=True):
    ckpt_dict = {"model_params": gaussians.capture(), "first_iter": iteration, "train_idx": scene.train_idxs, "base_iter": base_iter}

    if save_last:
        last_path = scene.model_path + "/last.pth"
        print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, last_path))
        torch.save(ckpt_dict, last_path)   

    if save_path is None:
        save_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
    print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, save_path))
    torch.save(ckpt_dict, save_path)   

def load_checkpoint(ckpt_path: str, gaussians, scene, opt, ignore_train_idxs=False):
    ckpt_dict = torch.load(ckpt_path)
    (model_params, first_iter, train_idxs) = ckpt_dict["model_params"], ckpt_dict["first_iter"], ckpt_dict["train_idx"]
    gaussians.restore(model_params, opt)
    if not ignore_train_idxs:
        scene.train_idxs = train_idxs

    base_iter = ckpt_dict.get("base_iter", 0)
    return first_iter, base_iter


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    base_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)


    override_train_idxs = override_train_idxs_dict.get(args.override_idxs, None)
    override_test_idxs = override_test_idxs_dict.get(args.override_idxs, None)

    scene = Scene(dataset, gaussians, random_init_pcd=args.random_init_pcd, llffhold=args.llffhold, override_train_idxs=override_train_idxs, override_test_idxs=override_test_idxs)
    gaussians.training_setup(opt)
    
    # Active View Selection
    schema = schema_dict[args.schema](dataset_size=len(scene.getTrainCameras()), scene=scene)
    print(f"schema: {schema.load_its}")
    scene.train_idxs = schema.init_views

    if args.inflate_path:
        scene.load_inflated_cameras(args.inflate_path, args.inflate_skip)
    
    active_method = methods_dict[args.method](args)

    init_ckpt_path = f"{args.model_path}/init.ckpt"
    if checkpoint:
        if os.path.exists(checkpoint):
            first_iter, base_iter = load_checkpoint(checkpoint, gaussians, scene, opt)
        else:
            print(f"[WARNING] checkpoint {checkpoint} doesn't exist, training from scratch")

    if args.scramble and first_iter == 0: # maybe init_ckpt has been save if preempted
        save_checkpoint(gaussians, first_iter, scene, base_iter, save_path=init_ckpt_path, save_last=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"train_idxs: {scene.train_idxs}")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        num_views = schema.num_views_to_add(iteration)
        if num_views > 0:
            try:
                # For sectioned training
                candidate_views_filter = getattr(schema, "candidate_views_filter")[iteration] if hasattr(schema, "candidate_views_filter") else None
                scene.candidate_views_filter = candidate_views_filter
                
                # Because selection is time consumeing
                selected_views = active_method.nbvs(gaussians, scene, num_views, pipe, background, exit_func=csm.should_exit)
            except RuntimeError as e:
                print(e)
                print("selector exited early")
                # NOTE: we use iteration - 1 because the selector is not done
                save_checkpoint(gaussians, iteration - 1, scene)
                csm.requeue()

            print(f"ITER {iteration}: selected views: {selected_views}")
            scene.train_idxs.extend(selected_views)
            print(f"ITER {iteration}: training views after selection: {scene.train_idxs}")

            gaussians.optimizer.zero_grad(set_to_none = True)

            if args.add_random_points > 0:
                gaussians.add_random_gaussians(args.add_random_points, scene.cameras_extent, expand_ratio=args.expand_ratio)

            if args.scramble:
                first_iter, _ = load_checkpoint(init_ckpt_path, gaussians, scene, opt, ignore_train_idxs=True)
                base_iter = iteration - 1
            else:
                gaussians.reset_opacity()

        iter_start.record()

        gaussians.update_learning_rate(iteration - base_iter)

        # Every 5000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        if iteration > args.sh_up_after and iteration % args.sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # We save before logging
        if csm.should_exit():
            save_checkpoint(gaussians, iteration - 1, scene)
            csm.requeue()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            before_selection = schema.num_views_to_add(iteration + 1) > 0
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), before_selection=before_selection)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            cur_iter = iteration - base_iter
            if cur_iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if cur_iter > opt.densify_from_iter and cur_iter % opt.densification_interval == 0:
                    size_threshold = 20 if cur_iter > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, args.min_opacity, scene.cameras_extent, size_threshold)
                
                if cur_iter % opt.opacity_reset_interval == 0 or (dataset.white_background and cur_iter == opt.densify_from_iter):
                    print(f"\nreset_opacity at {cur_iter}, base_iter")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
        if (iteration in checkpoint_iterations):
            save_checkpoint(gaussians, iteration, scene)
    wandb.finish()

        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, before_selection=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or before_selection:
        print(f"Running evaluation for iteration: {iteration}")
        torch.cuda.empty_cache()
        lpips = lpips_func("cuda", net_type='vgg')
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(idx), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(idx), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips.to(image.device)
                    lpips_test += lpips(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                log_dict = {config['name'] + '/l1_loss': l1_test, config['name'] + '/psnr': psnr_test,
                            config['name'] + '/ssim': ssim_test, config['name'] + '/lpips': psnr_test,}
                wandb.log(log_dict, step=iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()

import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--method", type=str, default="rand")
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--no_neg", action="store_true")
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--add_random_points", type=int, default=-1, help="add random points after each selection")
    parser.add_argument("--expand_ratio", type=float, default=2.0, help="expand raito to add random points")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=[])
    parser.add_argument("--random_init_pcd", type=int, default=-1, help="Use random points as initialization to train on Colmap dataset")
    parser.add_argument("--H_clamp_fraction", type=float, default=-1, help="fraction of H as upper bound, by defaul we don't clamp it")
    parser.add_argument("--H_max", type=float, default=-1, help="maximum value of the hessians")
    parser.add_argument("--scramble", action="store_true", help="scramble the checkpoint after selection")
    parser.add_argument("--inflate_path", default="", type=str, help="path to the inflated dataset")
    parser.add_argument("--inflate_skip", default=4, type=int, help="load 1/N inflated images")
    parser.add_argument("--llffhold", default=8, type=int, help="hold out Nth view when loading llff like dataset")
    parser.add_argument("--override_idxs", default=None, type=str, help="speical test idxs on uncertainty evaluation")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    
    if args.start_checkpoint is None:
        args.start_checkpoint = args.model_path + "/last.pth"
    
    print("Optimizing " + args.model_path)

    wandb.init(project='active', resume="allow", id=os.path.split(args.model_path.rstrip('/'))[-1], config=vars(args))

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    args.port = find_free_port()
    print(f"GUI at: {args.ip}:{args.port}")


    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             args)

    # All done
    print("\nTraining complete.")
