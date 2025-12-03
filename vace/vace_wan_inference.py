# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from annotators.utils import get_annotator

# Example prompts for different model versions, used when no prompt is provided by the user.
EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}

def validate_args(args):
    """
    Validates the command-line arguments and sets default values if necessary.
    """
    # Basic check for required arguments.
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # Set default sampling steps based on the task type.
    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    # Set default sampling shift factor.
    if args.sample_shift is None:
        args.sample_shift = 16

    # Set default number of frames.
    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 81

    # Set a random seed if not provided.
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Check if the specified size is supported for the selected model.
    assert args.size in SUPPORTED_SIZES[
        args.model_name], f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    return args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For I2V tasks, the aspect ratio follows the input image.")
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample. The number should be 4n+1.")
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='models/Wan2.1-VACE-1.3B/',
        help="The path to the model checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each forward pass to reduce GPU memory usage.")
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of Ulysses sequence parallelism in the DiT model.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of Ring Attention parallelism in the DiT model.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use Fully Sharded Data Parallel (FSDP) for the T5 model.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place the T5 model on the CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use Fully Sharded Data Parallel (FSDP) for the DiT model.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the generated output.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The specific file path to save the generated output.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="Path to the source video file.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="Path to the source mask file.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="Comma-separated list of source reference image paths.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The text prompt to guide generation.")
    parser.add_argument(
        "--use_prompt_extend",
        default='plain',
        choices=['plain', 'wan_zh', 'wan_en', 'wan_zh_ds', 'wan_en_ds'],
        help="Whether to use a prompt extender to enrich the input prompt.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=2025,
        help="The seed for random number generation.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used for the sampling process.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="Number of sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow-matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.")
    return parser


def _init_logging(rank):
    """Initializes logging for distributed training."""
    if rank == 0:
        # Set a verbose format for the main process.
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        # Suppress logging for other processes.
        logging.basicConfig(level=logging.ERROR)


def main(args):
    """Main function to run the Wan inference pipeline."""
    # Convert args to a Namespace object if it's a dict.
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    # --- Distributed Environment Setup ---
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    # Automatically decide whether to offload the model based on world size.
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    # Initialize the distributed process group if running on multiple GPUs.
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        # Assertions for single-GPU mode.
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    # --- Model Parallelism Setup (for large models) ---
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # --- Prompt Handling ---
    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        prompt_expander = get_annotator(config_type='prompt', config_task=args.use_prompt_extend, return_dict=False)

    # Load model-specific configuration.
    cfg = WAN_CONFIGS[args.model_name]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    # Synchronize the base seed across all processes.
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Use example prompt and inputs if none are provided.
    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.model_name]["prompt"]
        args.src_video = EXAMPLE_PROMPT[args.model_name].get("src_video", None)
        args.src_mask = EXAMPLE_PROMPT[args.model_name].get("src_mask", None)
        args.src_ref_images = EXAMPLE_PROMPT[args.model_name].get("src_ref_images", None)

    logging.info(f"Input prompt: {args.prompt}")
    # Extend the prompt if requested.
    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt = prompt_expander.forward(args.prompt)
            logging.info(f"Prompt extended from '{args.prompt}' to '{prompt}'")
            input_prompt = [prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    # --- Pipeline Initialization and Execution ---
    logging.info("Creating WanT2V pipeline.")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    # Prepare source video, mask, and reference images.
    src_video, src_mask, src_ref_images = wan_vace.prepare_source([args.src_video],
                                                                  [args.src_mask],
                                                                  [None if args.src_ref_images is None else args.src_ref_images.split(',')],
                                                                  args.frame_num, SIZE_CONFIGS[args.size], device)

    logging.info(f"Generating video...")
    # Run the main generation process.
    video = wan_vace.generate(
        args.prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    # --- Save Outputs ---
    ret_data = {}
    if rank == 0:
        # Create the output directory if it doesn't exist.
        if args.save_dir is None:
            save_dir = os.path.join('results', args.model_name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the generated video.
        if args.save_file is not None:
            save_file = args.save_file
        else:
            save_file = os.path.join(save_dir, 'out_video.mp4')
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logging.info(f"Saving generated video to {save_file}")
        ret_data['out_video'] = save_file

        # Save the source video for reference.
        save_file = os.path.join(save_dir, 'src_video.mp4')
        cache_video(
            tensor=src_video[0][None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logging.info(f"Saving src_video to {save_file}")
        ret_data['src_video'] = save_file

        # Save the source mask for reference.
        save_file = os.path.join(save_dir, 'src_mask.mp4')
        cache_video(
            tensor=src_mask[0][None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(0, 1))
        logging.info(f"Saving src_mask to {save_file}")
        ret_data['src_mask'] = save_file

        # Save the source reference images.
        if src_ref_images[0] is not None:
            for i, ref_img in enumerate(src_ref_images[0]):
                save_file = os.path.join(save_dir, f'src_ref_image_{i}.png')
                cache_image(
                    tensor=ref_img[:, 0, ...],
                    save_file=save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                logging.info(f"Saving src_ref_image_{i} to {save_file}")
                ret_data[f'src_ref_image_{i}'] = save_file
    logging.info("Finished.")
    return ret_data


if __name__ == "__main__":
    # Entry point of the script.
    args = get_parser().parse_args()
    main(args)
