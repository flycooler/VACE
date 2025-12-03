# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict

# English: This section defines common preprocessing configurations that can be reused across different pipelines.
# Chinese: 本节定义了可在不同流水线中重复使用的通用预处理配置。

######################### Common #########################
#------------------------ image ------------------------#
# English: A simple pass-through for an image.
# Chinese: 对图像进行简单的直通处理。
image_plain_anno = EasyDict()
image_plain_anno.NAME = "PlainImageAnnotator"
image_plain_anno.INPUTS = {"image": None}
image_plain_anno.OUTPUTS = {"image": None}

# English: A simple pass-through for an image mask.
# Chinese: 对图像蒙版进行简单的直通处理。
image_mask_plain_anno = EasyDict()
image_mask_plain_anno.NAME = "PlainMaskAnnotator"
image_mask_plain_anno.INPUTS = {"mask": None}
image_mask_plain_anno.OUTPUTS = {"mask": None}

# English: A simple pass-through for an augmented image mask.
# Chinese: 对增强后的图像蒙版进行简单的直通处理。
image_maskaug_plain_anno = EasyDict()
image_maskaug_plain_anno.NAME = "PlainMaskAugAnnotator"
image_maskaug_plain_anno.INPUTS = {"mask": None}
image_maskaug_plain_anno.OUTPUTS = {"mask": None}

# English: Inverts the colors of an augmented image mask.
# Chinese: 对增强后的图像蒙版进行颜色反转。
image_maskaug_invert_anno = EasyDict()
image_maskaug_invert_anno.NAME = "PlainMaskAugInvertAnnotator"
image_maskaug_invert_anno.INPUTS = {"mask": None}
image_maskaug_invert_anno.OUTPUTS = {"mask": None}

# English: Applies a specified augmentation to an image mask. The augmentation details are passed in `mask_cfg`.
# Chinese: 对图像蒙版应用指定的增强。增强的细节通过 `mask_cfg` 传入。
image_maskaug_anno = EasyDict()
image_maskaug_anno.NAME = "MaskAugAnnotator"
image_maskaug_anno.INPUTS = {"mask": None, 'mask_cfg': None}
image_maskaug_anno.OUTPUTS = {"mask": None}

# English: Draws a mask based on an image, bounding box, or other mode.
# Chinese: 根据图像、边界框或其他模式绘制蒙版。
image_mask_draw_anno = EasyDict()
image_mask_draw_anno.NAME = "MaskDrawAnnotator"
image_mask_draw_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_mask_draw_anno.OUTPUTS = {"mask": None}

# English: Creates a canvas and places the masked region randomly with augmentation.
# Chinese: 创建一个画布，并随机放置带有增强效果的蒙版区域。
image_maskaug_region_random_anno = EasyDict()
image_maskaug_region_random_anno.NAME = "RegionCanvasAnnotator"
image_maskaug_region_random_anno.SCALE_RANGE = [ 0.5, 1.0 ]
image_maskaug_region_random_anno.USE_AUG = True
image_maskaug_region_random_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_maskaug_region_random_anno.OUTPUTS = {"mask": None}

# English: Crops the masked region without placing it on a new canvas.
# Chinese: 裁剪蒙版区域，但不将其放置到新的画布上。
image_maskaug_region_crop_anno = EasyDict()
image_maskaug_region_crop_anno.NAME = "RegionCanvasAnnotator"
image_maskaug_region_crop_anno.SCALE_RANGE = [ 0.5, 1.0 ]
image_maskaug_region_crop_anno.USE_AUG = True
image_maskaug_region_crop_anno.USE_RESIZE = False
image_maskaug_region_crop_anno.USE_CANVAS = False
image_maskaug_region_crop_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_maskaug_region_crop_anno.OUTPUTS = {"mask": None}


#------------------------ video ------------------------#
# English: A simple pass-through for video frames.
# Chinese: 对视频帧进行简单的直通处理。
video_plain_anno = EasyDict()
video_plain_anno.NAME = "PlainVideoAnnotator"
video_plain_anno.INPUTS = {"frames": None}
video_plain_anno.OUTPUTS = {"frames": None}

# English: A simple pass-through for video masks.
# Chinese: 对视频蒙版进行简单的直通处理。
video_mask_plain_anno = EasyDict()
video_mask_plain_anno.NAME = "PlainMaskVideoAnnotator"
video_mask_plain_anno.INPUTS = {"masks": None}
video_mask_plain_anno.OUTPUTS = {"masks": None}

# English: A simple pass-through for augmented video masks.
# Chinese: 对增强后的视频蒙版进行简单的直通处理。
video_maskaug_plain_anno = EasyDict()
video_maskaug_plain_anno.NAME = "PlainMaskAugVideoAnnotator"
video_maskaug_plain_anno.INPUTS = {"masks": None}
video_maskaug_plain_anno.OUTPUTS = {"masks": None}

# English: Inverts the colors of augmented video masks.
# Chinese: 对增强后的视频蒙版进行颜色反转。
video_maskaug_invert_anno = EasyDict()
video_maskaug_invert_anno.NAME = "PlainMaskAugInvertVideoAnnotator"
video_maskaug_invert_anno.INPUTS = {"masks": None}
video_maskaug_invert_anno.OUTPUTS = {"masks": None}

# English: Expands the video masks, typically making them larger.
# Chinese: 扩展视频蒙版，通常使其变大。
video_mask_expand_anno = EasyDict()
video_mask_expand_anno.NAME = "ExpandMaskVideoAnnotator"
video_mask_expand_anno.INPUTS = {"masks": None}
video_mask_expand_anno.OUTPUTS = {"masks": None}

# English: Applies a specified augmentation to a video mask.
# Chinese: 对视频蒙版应用指定的增强。
video_maskaug_anno = EasyDict()
video_maskaug_anno.NAME = "MaskAugAnnotator"
video_maskaug_anno.INPUTS = {"mask": None, 'mask_cfg': None}
video_maskaug_anno.OUTPUTS = {"mask": None}

# English: Creates a layout mask for video, potentially using RAM tags for color coding.
# Chinese: 为视频创建布局蒙版，可能使用 RAM 标签进行颜色编码。
video_maskaug_layout_anno = EasyDict()
video_maskaug_layout_anno.NAME = "LayoutMaskAnnotator"
video_maskaug_layout_anno.RAM_TAG_COLOR_PATH = "models/VACE-Annotators/layout/ram_tag_color_list.txt"
video_maskaug_layout_anno.USE_AUG = True
video_maskaug_layout_anno.INPUTS = {"mask": None, 'mask_cfg': None}
video_maskaug_layout_anno.OUTPUTS = {"mask": None}


#------------------------ prompt ------------------------#
# English: A simple pass-through for a text prompt.
# Chinese: 对文本提示进行简单的直通处理。
prompt_plain_anno = EasyDict()
prompt_plain_anno.NAME = "PlainPromptAnnotator"
prompt_plain_anno.INPUTS = {"prompt": None}
prompt_plain_anno.OUTPUTS = {"prompt": None}
