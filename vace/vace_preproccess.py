# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import copy
import time
import inspect
import argparse
import importlib

from configs import VACE_PREPROCCESS_CONFIGS
import annotators
from annotators.utils import read_image, read_mask, read_video_frames, save_one_video, save_one_image


def parse_bboxes(s):
    """
    Parse bounding box strings.
    Input string format: "x1,y1,x2,y2 x1,y1,x2,y2 ..."
    Each bounding box consists of 4 comma-separated floats, and different bounding boxes are separated by spaces.
    """
    bboxes = []
    for bbox_str in s.split():
        coords = list(map(float, bbox_str.split(',')))
        if len(coords) != 4:
            raise ValueError(f"The bounding box requires 4 values, but the input is {len(coords)}.")
        bboxes.append(coords)
    return bboxes

def validate_args(args):
    """Validate the command line arguments."""
    # Check if the task is in the supported configurations
    assert args.task in VACE_PREPROCCESS_CONFIGS, f"Unsupport task: [{args.task}]"
    # Ensure at least one input type (video, image, or bbox) is provided
    assert args.video is not None or args.image is not None or args.bbox is not None, "Please specify the video or image or bbox."
    return args

def get_parser():
    parser = argparse.ArgumentParser(
        description="Data processing carried out by VACE"
    )
    parser.add_argument(
        "--task",
        type=str,
        default='',
        choices=list(VACE_PREPROCCESS_CONFIGS.keys()),
        help="The preprocessing task to run.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path of the videos to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path of the images to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="The specific mode of the task, such as firstframe, mask, bboxtrack, label...")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path of the mask images to be processed, separated by commas if there are multiple.")
    parser.add_argument(
        "--bbox",
        type=parse_bboxes,
        default=None,
        help="Enter the bounding box, with each four numbers separated by commas (x1, y1, x2, y2), and each pair separated by a space."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Enter the label to be processed, separated by commas if there are multiple."
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Enter the caption to be processed."
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="The direction of outpainting, any combination of left, right, up, down, with multiple directions separated by commas.")
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=None,
        help="The outward expansion ratio for outpainting.")
    parser.add_argument(
        "--expand_num",
        type=int,
        default=None,
        help="The number of frames to be extended by the video extension task.")
    parser.add_argument(
        "--maskaug_mode",
        type=str,
        default=None,
        help="The mode of mask augmentation, e.g., original, original_expand, hull, hull_expand, bbox, bbox_expand.")
    parser.add_argument(
        "--maskaug_ratio",
        type=float,
        default=None,
        help="The ratio for mask augmentation.")
    parser.add_argument(
        "--pre_save_dir",
        type=str,
        default=None,
        help="The path to save the preprocessed data.")
    parser.add_argument(
        "--save_fps",
        type=int,
        default=16,
        help="The fps for saving the video.")
    return parser


def preproccess():
    # Reserved pre-processing hook, currently not in use.
    pass

def proccess():
    # Reserved processing hook, currently not in use.
    pass

def postproccess():
    # Reserved post-processing hook, currently not in use.
    pass

def _prepare_input_data(args, input_params):
    """Prepare the data dictionary based on input parameters."""
    input_data = copy.deepcopy(input_params)
    video_paths = args.video.split(',') if args.video else []
    image_paths = args.image.split(',') if args.image else []

    # Helper function to read a video file
    def read_video_helper(video_path):
        frames, fps, width, height, num_frames = read_video_frames(video_path, use_type='cv2', info=True)
        assert frames is not None, f"Video read error: {video_path}"
        return frames, fps

    if 'video' in input_params or 'frames' in input_params:
        assert video_paths, "Please set video or check configs"
        # Read the first video
        frames, fps = read_video_helper(video_paths[0])
        if 'frames' in input_params:
            input_data['frames'] = frames
        if 'video' in input_params:
            input_data['video'] = args.video

    if 'frames_2' in input_params and len(video_paths) >= 2:
        # Read the second video if required
        frames_2, _ = read_video_helper(video_paths[1])
        input_data['frames_2'] = frames_2

    # Helper function to read an image file
    def read_image_helper(image_path):
        image, _, _ = read_image(image_path, use_type='pil', info=True)
        assert image is not None, f"Image read error: {image_path}"
        return image

    if 'image' in input_params:
        assert image_paths, "Please set image or check configs"
        # Read the first image
        input_data['image'] = read_image_helper(image_paths[0])

    if 'image_2' in input_params and len(image_paths) >= 2:
        # Read the second image if required
        input_data['image_2'] = read_image_helper(image_paths[1])

    if 'images' in input_params:
        assert image_paths, "Please set image or check configs"
        # Read all specified images
        input_data['images'] = [read_image_helper(p) for p in image_paths]

    if 'mask' in input_params and args.mask:
        # Read the mask file
        mask, _, _ = read_mask(args.mask.split(",")[0], use_type='pil', info=True)
        assert mask is not None, "Mask read error"
        input_data['mask'] = mask

    # Simplify assignment of other parameters
    arg_map = {'bbox': 'bbox', 'label': 'label', 'caption': 'caption', 'mode': 'mode', 'direction': 'direction', 'expand_ratio': 'expand_ratio', 'expand_num': 'expand_num'} # yapf: disable
    for param, arg_name in arg_map.items():
        if param in input_params and getattr(args, arg_name) is not None:
            value = getattr(args, arg_name)
            if isinstance(value, str) and ',' in value and param in ['label', 'direction']:
                value = value.split(',')
            elif param == 'bbox' and isinstance(value, list) and len(value) == 1:
                value = value[0]
            input_data[param] = value

    if 'mask_cfg' in input_params and args.maskaug_mode is not None:
        # Construct mask augmentation config
        mask_cfg = {"mode": args.maskaug_mode}
        if args.maskaug_ratio is not None:
            mask_cfg["kwargs"] = {'expand_ratio': args.maskaug_ratio, 'expand_iters': 5}
        input_data['mask_cfg'] = mask_cfg

    return input_data, locals().get('fps', None)

def _save_results(results, output_params, pre_save_dir, task_name, save_fps):
    """Save the processing results."""
    ret_data = {}
    
    # Helper function to save a video
    def save_video(data, path_suffix, key_name):
        if data is not None:
            save_path = os.path.join(pre_save_dir, f'{path_suffix}-{task_name}.mp4')
            save_one_video(save_path, data, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data[key_name] = save_path

    # Helper function to save a single image
    def save_image(data, path_suffix, key_name=None):
        if data is not None:
            save_path = os.path.join(pre_save_dir, f'{path_suffix}-{task_name}.png')
            save_one_image(save_path, data, use_type='pil')
            print(f"Save image result to {save_path}")
            if key_name:
                ret_data[key_name] = save_path

    # Map output parameters to saving logic
    output_map = {
        'frames': ('src_video', lambda r: r['frames'] if isinstance(r, dict) else r, 'src_video'),
        'masks': ('src_mask', lambda r: r['masks'] if isinstance(r, dict) else r, 'src_mask'),
        'image': ('src_ref_image', lambda r: r['image'] if isinstance(r, dict) else r, 'src_ref_images'),
        'mask': ('src_mask', lambda r: r['mask'] if isinstance(r, dict) else r, None),
    }

    # Iterate through the map and save results based on output parameters
    for param, (suffix, data_extractor, ret_key) in output_map.items():
        if param in output_params:
            data = data_extractor(results)
            if param in ['frames', 'masks']:
                save_video(data, suffix, ret_key)
            else:
                save_image(data, suffix, ret_key)

    if 'images' in output_params:
        # Special handling for saving multiple images
        ret_images = results.get('images') if isinstance(results, dict) else results
        if ret_images:
            saved_paths = []
            for i, img in enumerate(ret_images):
                if img:
                    save_path = os.path.join(pre_save_dir, f'src_ref_image_{i}-{task_name}.png')
                    save_one_image(save_path, img, use_type='pil')
                    print(f"Save image result to {save_path}")
                    saved_paths.append(save_path)
            if saved_paths:
                ret_data['src_ref_images'] = ','.join(saved_paths)

    return ret_data

def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    task_name = args.task

    # init class
    task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)[task_name]
    class_name = task_cfg.pop("NAME")
    input_params = task_cfg.pop("INPUTS")
    output_params = task_cfg.pop("OUTPUTS")

    # input data
    input_data, fps = _prepare_input_data(args, input_params)

    # processing
    pre_ins = getattr(annotators, class_name)(cfg=task_cfg, device=f'cuda:{os.getenv("RANK", 0)}')
    results = pre_ins.forward(**input_data)

    # output data
    save_fps = fps if fps is not None else args.save_fps
    if args.pre_save_dir is None:
        pre_save_dir = os.path.join('processed', task_name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    else:
        pre_save_dir = args.pre_save_dir
    if not os.path.exists(pre_save_dir):
        os.makedirs(pre_save_dir)

    ret_data = _save_results(results, output_params, pre_save_dir, task_name, save_fps)
    return ret_data


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
