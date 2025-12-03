# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from einops import rearrange

from .utils import convert_to_numpy, resize_image, resize_image_ori

class DepthAnnotator:
    # English: An annotator for single-image depth estimation using the MiDaS model.
    # Chinese: 使用 MiDaS 模型进行单张图像深度估计的处理器。
    def __init__(self, cfg, device=None):
        from .midas.api import MiDaSInference
        pretrained_model = cfg['PRETRAINED_MODEL']
        # English: Set the device for computation (CUDA if available, otherwise CPU).
        # Chinese: 设置计算设备（如果可用则使用CUDA，否则使用CPU）。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # English: Load the MiDaS model.
        # Chinese: 加载 MiDaS 模型。
        self.model = MiDaSInference(model_type='dpt_hybrid', model_path=pretrained_model).to(self.device)
        self.a = cfg.get('A', np.pi * 2.0)
        self.bg_th = cfg.get('BG_TH', 0.1)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        # English: Convert the input to a NumPy array.
        # Chinese: 将输入转换为 NumPy 数组。
        image = convert_to_numpy(image)
        image_depth = image
        h, w, c = image.shape
        # English: Resize the image for model input and get the resize factor.
        # Chinese: 为模型输入调整图像大小，并获取缩放因子。
        image_depth, k = resize_image(image_depth,
                                      1024 if min(h, w) > 1024 else min(h, w))
        # English: Preprocess the image: convert to tensor, normalize, and add a batch dimension.
        # Chinese: 预处理图像：转换为张量、归一化并添加批次维度。
        image_depth = torch.from_numpy(image_depth).float().to(self.device)
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        # English: Perform inference to get the raw depth map.
        # Chinese: 执行推理以获取原始深度图。
        depth = self.model(image_depth)[0]

        # English: Normalize the depth map to the range [0, 1].
        # Chinese: 将深度图归一化到 [0, 1] 范围。
        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        # English: Convert the normalized depth map to a visualizable 8-bit image.
        # Chinese: 将归一化后的深度图转换为可视化的8位图像。
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        # English: Convert the single-channel depth image to a 3-channel grayscale image.
        # Chinese: 将单通道深度图转换为三通道灰度图像。
        depth_image = depth_image[..., None].repeat(3, 2)

        # English: Resize the depth image back to the original input dimensions.
        # Chinese: 将深度图像调整回原始输入尺寸。
        depth_image = resize_image_ori(h, w, depth_image, k)
        return depth_image


class DepthVideoAnnotator(DepthAnnotator):
    # English: A wrapper class to apply the DepthAnnotator to each frame of a video.
    # Chinese: 一个包装类，用于将 DepthAnnotator 应用于视频的每一帧。
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            # English: Process each frame individually using the parent class's forward method.
            # Chinese: 使用父类的 forward 方法单独处理每一帧。
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames


class DepthV2Annotator:
    # English: An annotator for single-image depth estimation using the DepthAnythingV2 model.
    # Chinese: 使用 DepthAnythingV2 模型进行单张图像深度估计的处理器。
    def __init__(self, cfg, device=None):
        from .depth_anything_v2.dpt import DepthAnythingV2
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # English: Load the DepthAnythingV2 model.
        # Chinese: 加载 DepthAnythingV2 模型。
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                pretrained_model,
                map_location=self.device
            )
        )
        self.model.eval()

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        # English: Convert input to NumPy array and perform inference.
        # Chinese: 将输入转换为 NumPy 数组并执行推理。
        image = convert_to_numpy(image)
        depth = self.model.infer_image(image)

        # English: Normalize and convert the raw depth output to a visualizable 8-bit image.
        # Chinese: 将原始深度输出归一化并转换为可视化的8位图像。
        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        # English: Convert to a 3-channel grayscale image.
        # Chinese: 转换为三通道灰度图像。
        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return depth_image


class DepthV2VideoAnnotator(DepthV2Annotator):
    # English: A wrapper class to apply the DepthV2Annotator to each frame of a video.
    # Chinese: 一个包装类，用于将 DepthV2Annotator 应用于视频的每一帧。
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            # English: Process each frame individually.
            # Chinese: 单独处理每一帧。
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames
