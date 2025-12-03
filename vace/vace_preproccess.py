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
    # English: Parse bounding box strings.
    # Chinese: 解析边界框字符串。
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
    """
    # English: Validate the command line arguments.
    # Chinese: 验证命令行参数。
    """
    # Check if the task is in the supported configurations
    assert args.task in VACE_PREPROCCESS_CONFIGS, f"Unsupport task: [{args.task}]"
    # English: Ensure at least one input type (video, image, or bbox) is provided.
    # Chinese: 确保至少提供了一种输入类型（视频、图像或边界框）。
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
        help="The preprocessing task to run. (要运行的预处理任务。)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path of the videos to be processed, separated by commas if there are multiple. (要处理的视频路径，多个用逗号分隔。)")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path of the images to be processed, separated by commas if there are multiple. (要处理的图像路径，多个用逗号分隔。)")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="The specific mode of the task, such as firstframe, mask, bboxtrack, label... (任务的特定模式，例如 firstframe, mask, bboxtrack, label...)")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path of the mask images to be processed, separated by commas if there are multiple. (要处理的蒙版图像路径，多个用逗号分隔。)")
    parser.add_argument(
        "--bbox",
        type=parse_bboxes,
        default=None,
        help="Enter the bounding box, with each four numbers separated by commas (x1, y1, x2, y2), and each pair separated by a space. (输入边界框，每四个数字由逗号分隔 (x1, y1, x2, y2)，每对由空格分隔。)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Enter the label to be processed, separated by commas if there are multiple. (输入要处理的标签，多个用逗号分隔。)"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Enter the caption to be processed. (输入要处理的文本描述。)"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="The direction of outpainting, any combination of left, right, up, down, with multiple directions separated by commas. (图像外扩的方向，可以是 left, right, up, down 的任意组合，多个用逗号分隔。)")
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=None,
        help="The outward expansion ratio for outpainting. (图像外扩的比例。)")
    parser.add_argument(
        "--expand_num",
        type=int,
        default=None,
        help="The number of frames to be extended by the video extension task. (视频扩展任务要扩展的帧数。)")
    parser.add_argument(
        "--maskaug_mode",
        type=str,
        default=None,
        help="The mode of mask augmentation, e.g., original, original_expand, hull, hull_expand, bbox, bbox_expand. (蒙版增强的模式。)")
    parser.add_argument(
        "--maskaug_ratio",
        type=float,
        default=None,
        help="The ratio for mask augmentation. (蒙版增强的比例。)")
    parser.add_argument(
        "--pre_save_dir",
        type=str,
        default=None,
        help="The path to save the preprocessed data. (保存预处理后数据的路径。)")
    parser.add_argument(
        "--save_fps",
        type=int,
        default=16,
        help="The fps for saving the video. (保存视频时的帧率。)")
    return parser


def preproccess():
    # English: Reserved pre-processing hook, currently not in use.
    # Chinese: 预留的预处理钩子函数，当前未使用。
    # Reserved pre-processing hook, currently not in use.
    pass

def proccess():
    # English: Reserved processing hook, currently not in use.
    # Chinese: 预留的处理钩子函数，当前未使用。
    # Reserved processing hook, currently not in use.
    pass

def postproccess():
    # Reserved post-processing hook, currently not in use.
    pass

def _prepare_input_data(args, input_params):
    """
    # English: Prepare the data dictionary based on input parameters.
    # Chinese: 根据输入参数准备数据字典。
    """
    input_data = copy.deepcopy(input_params)
    video_paths = args.video.split(',') if args.video else []
    image_paths = args.image.split(',') if args.image else []

    # English: Helper function to read a video file.
    # Chinese: 用于读取视频文件的辅助函数。
    # Helper function to read a video file
    def read_video_helper(video_path):
        frames, fps, width, height, num_frames = read_video_frames(video_path, use_type='cv2', info=True)
        assert frames is not None, f"Video read error: {video_path}"
        return frames, fps

    if 'video' in input_params or 'frames' in input_params:
        assert video_paths, "Please set video or check configs"
        # English: Read the first video.
        # Chinese: 读取第一个视频。
        # Read the first video
        frames, fps = read_video_helper(video_paths[0])
        if 'frames' in input_params:
            input_data['frames'] = frames
        if 'video' in input_params:
            input_data['video'] = args.video

    if 'frames_2' in input_params and len(video_paths) >= 2:
        # English: Read the second video if required.
        # Chinese: 如果需要，则读取第二个视频。
        # Read the second video if required
        frames_2, _ = read_video_helper(video_paths[1])
        input_data['frames_2'] = frames_2

    # English: Helper function to read an image file.
    # Chinese: 用于读取图像文件的辅助函数。
    # Helper function to read an image file
    def read_image_helper(image_path):
        image, _, _ = read_image(image_path, use_type='pil', info=True)
        assert image is not None, f"Image read error: {image_path}"
        return image

    if 'image' in input_params:
        assert image_paths, "Please set image or check configs"
        # English: Read the first image.
        # Chinese: 读取第一个图像。
        # Read the first image
        input_data['image'] = read_image_helper(image_paths[0])

    if 'image_2' in input_params and len(image_paths) >= 2:
        # English: Read the second image if required.
        # Chinese: 如果需要，则读取第二个图像。
        # Read the second image if required
        input_data['image_2'] = read_image_helper(image_paths[1])

    if 'images' in input_params:
        assert image_paths, "Please set image or check configs"
        # English: Read all specified images.
        # Chinese: 读取所有指定的图像。
        # Read all specified images
        input_data['images'] = [read_image_helper(p) for p in image_paths]

    if 'mask' in input_params and args.mask:
        # English: Read the mask file.
        # Chinese: 读取蒙版文件。
        # Read the mask file
        mask, _, _ = read_mask(args.mask.split(",")[0], use_type='pil', info=True)
        assert mask is not None, "Mask read error"
        input_data['mask'] = mask

    # English: Simplify assignment of other parameters.
    # Chinese: 简化其他参数的赋值。
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
        # English: Construct mask augmentation config.
        # Chinese: 构造蒙版增强配置。
        # Construct mask augmentation config
        mask_cfg = {"mode": args.maskaug_mode}
        if args.maskaug_ratio is not None:
            mask_cfg["kwargs"] = {'expand_ratio': args.maskaug_ratio, 'expand_iters': 5}
        input_data['mask_cfg'] = mask_cfg

    return input_data, locals().get('fps', None)

def _save_results(results, output_params, pre_save_dir, task_name, save_fps):
    """
    # English: Save the processing results.
    # Chinese: 保存处理结果。
    """
    ret_data = {}
    
    # English: Helper function to save a video.
    # Chinese: 用于保存视频的辅助函数。
    # Helper function to save a video
    def save_video(data, path_suffix, key_name):
        if data is not None:
            save_path = os.path.join(pre_save_dir, f'{path_suffix}-{task_name}.mp4')
            save_one_video(save_path, data, fps=save_fps)
            print(f"Save frames result to {save_path}")
            ret_data[key_name] = save_path

    # English: Helper function to save a single image.
    # Chinese: 用于保存单张图像的辅助函数。
    # Helper function to save a single image
    def save_image(data, path_suffix, key_name=None):
        if data is not None:
            save_path = os.path.join(pre_save_dir, f'{path_suffix}-{task_name}.png')
            save_one_image(save_path, data, use_type='pil')
            print(f"Save image result to {save_path}")
            if key_name:
                ret_data[key_name] = save_path

    # English: Map output parameters to saving logic.
    # Chinese: 将输出参数映射到保存逻辑。
    # Map output parameters to saving logic
    output_map = {
        'frames': ('src_video', lambda r: r['frames'] if isinstance(r, dict) else r, 'src_video'),
        'masks': ('src_mask', lambda r: r['masks'] if isinstance(r, dict) else r, 'src_mask'),
        'image': ('src_ref_image', lambda r: r['image'] if isinstance(r, dict) else r, 'src_ref_images'),
        'mask': ('src_mask', lambda r: r['mask'] if isinstance(r, dict) else r, None),
    }

    # English: Iterate through the map and save results based on output parameters.
    # Chinese: 遍历映射表，并根据输出参数保存结果。
    # Iterate through the map and save results based on output parameters
    for param, (suffix, data_extractor, ret_key) in output_map.items():
        if param in output_params:
            data = data_extractor(results)
            if param in ['frames', 'masks']:
                save_video(data, suffix, ret_key)
            else:
                save_image(data, suffix, ret_key)

    if 'images' in output_params:
        # English: Special handling for saving multiple images.
        # Chinese: 对保存多张图像进行特殊处理。
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
    # English: Convert args to a Namespace object if it's a dict.
    # Chinese: 如果 args 是字典，则将其转换为 Namespace 对象。
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    # English: Validate arguments.
    # Chinese: 验证参数。
    args = validate_args(args)

    task_name = args.task

    # English: 1. Initialization: Load configuration based on the task name.
    # Chinese: 1. 初始化：根据任务名称加载配置。
    # init class
    task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)[task_name]
    class_name = task_cfg.pop("NAME")
    input_params = task_cfg.pop("INPUTS")
    output_params = task_cfg.pop("OUTPUTS")

    # English: 2. Prepare Input Data: Call helper function to load and prepare data.
    # Chinese: 2. 准备输入数据：调用辅助函数加载和准备数据。
    # input data
    input_data, fps = _prepare_input_data(args, input_params)

    # English: 3. Processing: Dynamically instantiate the processor class and call its forward method.
    # Chinese: 3. 处理：动态实例化处理器类并调用其 forward 方法。
    # processing
    pre_ins = getattr(annotators, class_name)(cfg=task_cfg, device=f'cuda:{os.getenv("RANK", 0)}')
    results = pre_ins.forward(**input_data)

    # English: 4. Save Output Data.
    # Chinese: 4. 保存输出数据。
    # output data
    # English: Determine the FPS for saving the video.
    # Chinese: 确定保存视频的帧率。
    save_fps = fps if fps is not None else args.save_fps
    # English: Determine and create the save directory.
    # Chinese: 确定并创建保存目录。
    if args.pre_save_dir is None:
        pre_save_dir = os.path.join('processed', task_name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    else:
        pre_save_dir = args.pre_save_dir
    if not os.path.exists(pre_save_dir):
        os.makedirs(pre_save_dir)

    # English: Call helper function to save results and get return data.
    # Chinese: 调用辅助函数保存结果并获取返回数据。
    ret_data = _save_results(results, output_params, pre_save_dir, task_name, save_fps)
    return ret_data


if __name__ == "__main__":
    # English: When the script is run as the main program.
    # Chinese: 当脚本作为主程序运行时。
    args = get_parser().parse_args()
    main(args)
