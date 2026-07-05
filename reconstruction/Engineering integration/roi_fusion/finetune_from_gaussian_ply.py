#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from random import randint
from time import time

import torch
from tqdm import tqdm


def add_repo_to_path(repo: str) -> None:
    repo = os.path.abspath(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def main() -> int:
    parser = ArgumentParser(description="Short fine-tune a 3DGS model initialized from an existing Gaussian PLY.")
    parser.add_argument("--repo", required=True, help="Path to graphdeco gaussian-splatting repo")
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--init_ply", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", "-r", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--start_iteration", type=int, default=30000)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--position_lr", type=float, default=1.6e-6)
    parser.add_argument("--feature_lr", type=float, default=5e-4)
    parser.add_argument("--opacity_lr", type=float, default=5e-3)
    parser.add_argument("--scaling_lr", type=float, default=1e-3)
    parser.add_argument("--rotation_lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--plant_ply", default="", help="Optional manually segmented plant Gaussian PLY for weighted fine-tune")
    parser.add_argument("--plant_mask_path", default="", help="Optional .npz mask path. Created from --plant_ply when missing")
    parser.add_argument("--plant_match_tolerance", type=float, default=1e-6)
    parser.add_argument("--plant_weight_factor", type=float, default=1.0, help="Gradient multiplier for plant Gaussians")
    args = parser.parse_args()

    add_repo_to_path(args.repo)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from train import prepare_output_and_logger, training_report
    from utils.general_utils import get_expon_lr_func, safe_state
    from utils.loss_utils import l1_loss, ssim
    from plant_weighting import (
        apply_plant_gradient_weights,
        build_plant_mask,
        load_plant_mask,
        save_plant_mask,
    )
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

        sparse_adam_available = True
    except Exception:
        sparse_adam_available = False

    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)

    # Build repo-native parameter groups from a synthetic parser namespace.
    repo_parser = ArgumentParser()
    lp = ModelParams(repo_parser)
    op = OptimizationParams(repo_parser)
    pp = PipelineParams(repo_parser)
    repo_args = repo_parser.parse_args([])
    repo_args.source_path = args.source_path
    repo_args.model_path = args.model_path
    repo_args.images = args.images
    repo_args.depths = ""
    repo_args.resolution = args.resolution
    repo_args.white_background = args.white_background
    repo_args.train_test_exp = False
    repo_args.data_device = "cuda"
    repo_args.eval = args.eval
    repo_args.iterations = args.start_iteration + args.iterations
    repo_args.lambda_dssim = args.lambda_dssim
    repo_args.position_lr_init = args.position_lr
    repo_args.position_lr_final = args.position_lr
    repo_args.position_lr_delay_mult = 1.0
    repo_args.position_lr_max_steps = args.iterations
    repo_args.feature_lr = args.feature_lr
    repo_args.opacity_lr = args.opacity_lr
    repo_args.scaling_lr = args.scaling_lr
    repo_args.rotation_lr = args.rotation_lr
    repo_args.densify_until_iter = 0
    repo_args.densify_from_iter = 10**9
    repo_args.opacity_reset_interval = 10**9
    repo_args.random_background = False
    repo_args.optimizer_type = "default"
    repo_args.convert_SHs_python = False
    repo_args.compute_cov3D_python = False
    repo_args.debug = False
    repo_args.antialiasing = False

    tb_writer = prepare_output_and_logger(lp.extract(repo_args))
    gaussians = GaussianModel(repo_args.sh_degree, repo_args.optimizer_type)
    scene = Scene(lp.extract(repo_args), gaussians, load_iteration=None, shuffle=True)
    gaussians.load_ply(args.init_ply, False)
    gaussians.training_setup(op.extract(repo_args))
    plant_mask_tensor = None
    plant_mask_meta = None
    if args.plant_weight_factor != 1.0:
        if not args.plant_mask_path and not args.plant_ply:
            raise ValueError("--plant_weight_factor requires --plant_mask_path or --plant_ply")
        if args.plant_mask_path and os.path.exists(args.plant_mask_path):
            plant_mask, plant_mask_meta = load_plant_mask(args.plant_mask_path)
        else:
            if not args.plant_ply:
                raise ValueError("--plant_mask_path does not exist; pass --plant_ply to create it")
            plant_mask, plant_mask_meta = build_plant_mask(args.init_ply, args.plant_ply, args.plant_match_tolerance)
            output_mask_path = args.plant_mask_path or os.path.join(args.model_path, "plant_mask.npz")
            save_plant_mask(plant_mask, output_mask_path, plant_mask_meta)
            args.plant_mask_path = output_mask_path
        if plant_mask.shape[0] != gaussians.get_xyz.shape[0]:
            raise ValueError(
                f"plant mask length {plant_mask.shape[0]} does not match Gaussian count {gaussians.get_xyz.shape[0]}"
            )
        if int(plant_mask.sum()) == 0:
            raise ValueError("plant mask is empty; check --plant_ply and --plant_match_tolerance")
        plant_mask_tensor = torch.from_numpy(plant_mask).to(device=gaussians.get_xyz.device, dtype=torch.bool)
        print(
            f"Plant-weighted fine-tune enabled: {int(plant_mask.sum())}/{plant_mask.shape[0]} "
            f"Gaussians weighted by {args.plant_weight_factor}"
        )

    bg_color = [1, 1, 1] if repo_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(0.0, 0.0, max_steps=repo_args.iterations)
    _ = depth_l1_weight

    first_iter = args.start_iteration
    final_iter = args.start_iteration + args.iterations
    save_iterations = set(range(args.start_iteration + args.save_every, final_iter + 1, args.save_every))
    save_iterations.add(final_iter)
    test_iterations = {final_iter}

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    progress_bar = tqdm(range(first_iter, final_iter), desc="Fine-tuning progress")
    ema_loss = 0.0
    started = time()
    for iteration in range(first_iter + 1, final_iter + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pp.extract(repo_args),
            background,
            use_trained_exp=False,
            separate_sh=sparse_adam_available,
        )
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss_l1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - repo_args.lambda_dssim) * loss_l1 + repo_args.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        if plant_mask_tensor is not None:
            apply_plant_gradient_weights(
                [
                    gaussians._xyz,
                    gaussians._features_dc,
                    gaussians._features_rest,
                    gaussians._opacity,
                    gaussians._scaling,
                    gaussians._rotation,
                ],
                plant_mask_tensor,
                args.plant_weight_factor,
            )

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}"})
                progress_bar.update(10)
            if iteration in test_iterations:
                training_report(
                    tb_writer,
                    iteration,
                    loss_l1,
                    loss,
                    l1_loss,
                    0.0,
                    [iteration],
                    scene,
                    render,
                    (pp.extract(repo_args), background, 1.0, sparse_adam_available, None, False),
                    False,
                )
            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
            if iteration < final_iter:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()
    meta = {
        "init_ply": os.path.abspath(args.init_ply),
        "source_path": os.path.abspath(args.source_path),
        "model_path": os.path.abspath(args.model_path),
        "start_iteration": args.start_iteration,
        "iterations": args.iterations,
        "final_iteration": final_iter,
        "train_time_sec": time() - started,
        "densification": "disabled",
        "plant_weight_factor": args.plant_weight_factor,
        "plant_ply": os.path.abspath(args.plant_ply) if args.plant_ply else "",
        "plant_mask_path": os.path.abspath(args.plant_mask_path) if args.plant_mask_path else "",
        "plant_mask_meta": plant_mask_meta,
    }
    with open(os.path.join(args.model_path, "finetune_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
