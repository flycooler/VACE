# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .utils import convert_to_torch

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    # English: A standard residual block for deep neural networks.
    # Chinese: 一个用于深度神经网络的标准残差块。
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            # English: Reflection padding is used to reduce border artifacts.
            # Chinese: 使用反射填充以减少边界伪影。
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # English: The output is the sum of the input and the output of the convolutional block (skip connection).
        # Chinese: 输出是输入与卷积块输出的和（跳跃连接）。
        return x + self.conv_block(x)


class ContourInference(nn.Module):
    # English: The main image-to-image translation model for converting an image to a scribble/contour map.
    # Chinese: 用于将图像转换为涂鸦/轮廓图的主要图像到图像转换模型。
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(ContourInference, self).__init__()

        # Initial convolution block
        # English: The first layer to extract initial features from the input image.
        # Chinese: 从输入图像中提取初始特征的第一层。
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        # English: Reduces the spatial dimensions of the feature maps.
        # Chinese: 减小特征图的空间维度。
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        # English: The core of the network, for deep feature transformation.
        # Chinese: 网络的核心，用于进行深度特征变换。
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        # English: Increases the spatial dimensions back to the original size.
        # Chinese: 将空间维度恢复到原始尺寸。
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features,
                                   out_features,
                                   3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        # English: Generates the final output image. A sigmoid function is used to normalize the output to [0, 1].
        # Chinese: 生成最终的输出图像。使用 Sigmoid 函数将输出归一化到 [0, 1] 范围。
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        # English: Defines the forward pass through the network architecture (encoder -> residual blocks -> decoder).
        # Chinese: 定义通过网络架构（编码器 -> 残差块 -> 解码器）的前向传播路径。
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class ScribbleAnnotator:
    # English: The main annotator class for generating a scribble/line-art image from a single input image.
    # Chinese: 用于从单张输入图像生成涂鸦/线稿图的主处理器类。
    def __init__(self, cfg, device=None):
        input_nc = cfg.get('INPUT_NC', 3)
        output_nc = cfg.get('OUTPUT_NC', 1)
        n_residual_blocks = cfg.get('N_RESIDUAL_BLOCKS', 3)
        sigmoid = cfg.get('SIGMOID', True)
        pretrained_model = cfg['PRETRAINED_MODEL']
        # English: Set the computation device.
        # Chinese: 设置计算设备。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # English: Instantiate the ContourInference model.
        # Chinese: 实例化 ContourInference 模型。
        self.model = ContourInference(input_nc, output_nc, n_residual_blocks,
                                      sigmoid)
        # English: Load the pretrained weights.
        # Chinese: 加载预训练权重。
        self.model.load_state_dict(torch.load(pretrained_model, weights_only=True))
        # English: Set the model to evaluation mode and move it to the specified device.
        # Chinese: 将模型设置为评估模式并移动到指定设备。
        self.model = self.model.eval().requires_grad_(False).to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        # English: Determine if the input is a single image or a batch.
        # Chinese: 判断输入是单张图像还是一个批次。
        is_batch = False if len(image.shape) == 3 else True
        # English: Preprocess the image: convert to tensor, normalize, and move to device.
        # Chinese: 预处理图像：转换为张量、归一化并移动到设备。
        image = convert_to_torch(image)
        if len(image.shape) == 3:
            image = rearrange(image, 'h w c -> 1 c h w')
        image = image.float().div(255).to(self.device)
        # English: Perform inference with the model.
        # Chinese: 使用模型进行推理。
        contour_map = self.model(image)
        # English: Postprocess the output: scale to [0, 255], convert to numpy, and repeat channels to create a 3-channel image.
        # Chinese: 后处理输出：缩放到 [0, 255] 范围，转换为 numpy 数组，并重复通道以创建三通道图像。
        contour_map = (contour_map.squeeze(dim=1) * 255.0).clip(
            0, 255).cpu().numpy().astype(np.uint8)
        contour_map = contour_map[..., None].repeat(3, -1)
        if not is_batch:
            contour_map = contour_map.squeeze()
        return contour_map


class ScribbleVideoAnnotator(ScribbleAnnotator):
    # English: A wrapper class to apply the ScribbleAnnotator to each frame of a video.
    # Chinese: 一个包装类，用于将 ScribbleAnnotator 应用于视频的每一帧。
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            # English: Process each frame individually using the parent class's forward method.
            # Chinese: 使用父类的 forward 方法单独处理每一帧。
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames