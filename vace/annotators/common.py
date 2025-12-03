# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# English: A pass-through annotator for images. It simply returns the input image without any modification.
# Chinese: 图像的直通处理器。它仅返回输入的图像，不进行任何修改。
class PlainImageAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, image):
        # English: Returns the input image as is.
        # Chinese: 按原样返回输入的图像。
        return image

# English: A pass-through annotator for video frames. It returns the list of frames unmodified.
# Chinese: 视频帧的直通处理器。它不经修改地返回帧列表。
class PlainVideoAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, frames):
        # English: Returns the input frames as is.
        # Chinese: 按原样返回输入的视频帧。
        return frames

# English: A pass-through annotator for a single mask.
# Chinese: 单个蒙版的直通处理器。
class PlainMaskAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, mask):
        # English: Returns the input mask as is.
        # Chinese: 按原样返回输入的蒙版。
        return mask

# English: An annotator that inverts the colors of a single mask.
# Chinese: 用于反转单个蒙版颜色的处理器。
class PlainMaskAugInvertAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, mask):
        # English: Inverts the mask (assuming white is 255).
        # Chinese: 反转蒙版（假设白色为255）。
        return 255 - mask

# English: A pass-through annotator for an augmented mask.
# Chinese: 增强后蒙版的直通处理器。
class PlainMaskAugAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, mask):
        # English: Returns the input mask as is.
        # Chinese: 按原样返回输入的蒙版。
        return mask

# English: A pass-through annotator for a list of video masks.
# Chinese: 视频蒙版列表的直通处理器。
class PlainMaskVideoAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, mask):
        # English: Returns the input mask list as is.
        # Chinese: 按原样返回输入的蒙版列表。
        return mask

# English: A pass-through annotator for a list of augmented video masks.
# Chinese: 增强后视频蒙版列表的直通处理器。
class PlainMaskAugVideoAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, masks):
        # English: Returns the input mask list as is.
        # Chinese: 按原样返回输入的蒙版列表。
        return masks

# English: An annotator that inverts a batch of video masks.
# Chinese: 用于反转一批视频蒙版的处理器。
class PlainMaskAugInvertVideoAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, masks):
        # English: Inverts each mask in the list (assuming white is 255).
        # Chinese: 反转列表中的每一个蒙版（假设白色为255）。
        return [255 - mask for mask in masks]

# English: An annotator that expands a single mask into a video sequence by duplication.
# Chinese: 通过复制将单个蒙版扩展为视频序列的处理器。
class ExpandMaskVideoAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, mask, expand_num):
        # English: Returns a list containing the input mask repeated `expand_num` times.
        # Chinese: 返回一个列表，其中包含重复 `expand_num` 次的输入蒙版。
        return [mask] * expand_num

# English: A pass-through annotator for text prompts.
# Chinese: 文本提示的直通处理器。
class PlainPromptAnnotator:
    def __init__(self, cfg):
        # English: The constructor does not need to initialize any models.
        # Chinese: 构造函数无需初始化任何模型。
        pass
    def forward(self, prompt):
        # English: Returns the input prompt as is.
        # Chinese: 按原样返回输入的提示。
        return prompt