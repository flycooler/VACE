# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# English: This file serves as the public API for the 'annotators' package.
#          It imports all annotator classes from their respective modules,
#          making them directly accessible under the 'annotators' namespace.
# Chinese: 这个文件是 'annotators' 包的公共 API。
#          它从各自的模块中导入所有的 annotator 类，
#          使得它们可以在 'annotators' 命名空间下被直接访问。

# --- Control Signal Annotators ---
# English: Annotators for extracting structural or motion information like depth, pose, etc.
# Chinese: 用于提取结构或运动信息（如深度、姿态等）的控制信号处理器。
from .depth import DepthAnnotator, DepthVideoAnnotator, DepthV2VideoAnnotator
from .flow import FlowAnnotator, FlowVisAnnotator
from .gray import GrayAnnotator, GrayVideoAnnotator
from .pose import PoseBodyFaceAnnotator, PoseBodyFaceVideoAnnotator, PoseAnnotator, PoseBodyVideoAnnotator, PoseBodyAnnotator
from .scribble import ScribbleAnnotator, ScribbleVideoAnnotator
from .layout import LayoutBboxAnnotator, LayoutMaskAnnotator, LayoutTrackAnnotator

# --- Segmentation and Detection Annotators ---
# English: Annotators for identifying, segmenting, or tracking objects.
# Chinese: 用于识别、分割或跟踪物体的处理器。
from .gdino import GDINOAnnotator, GDINORAMAnnotator
from .ram import RAMAnnotator
from .salient import SalientAnnotator, SalientVideoAnnotator
from .sam import SAMImageAnnotator
from .sam2 import SAM2ImageAnnotator, SAM2VideoAnnotator, SAM2SalientVideoAnnotator, SAM2GDINOVideoAnnotator
from .face import FaceAnnotator
from .subject import SubjectAnnotator

# --- Video/Image Manipulation Annotators ---
# English: Annotators for editing, extending, or transforming images and videos.
# Chinese: 用于编辑、扩展或变换图像和视频的处理器。
from .frameref import FrameRefExtractAnnotator, FrameRefExpandAnnotator
from .inpainting import InpaintingAnnotator, InpaintingVideoAnnotator
from .outpainting import OutpaintingAnnotator, OutpaintingInnerAnnotator, OutpaintingVideoAnnotator, OutpaintingInnerVideoAnnotator
from .composition import CompositionAnnotator, ReferenceAnythingAnnotator, AnimateAnythingAnnotator, SwapAnythingAnnotator, ExpandAnythingAnnotator, MoveAnythingAnnotator

# --- Utility and Helper Annotators ---
# English: Common, reusable annotators for basic operations like masking, augmentation, and canvas manipulation.
# Chinese: 用于基本操作（如蒙版、增强和画布处理）的通用、可复用的处理器。
from .common import PlainImageAnnotator, PlainMaskAnnotator, PlainMaskAugAnnotator, PlainMaskVideoAnnotator, PlainVideoAnnotator, PlainMaskAugVideoAnnotator, PlainMaskAugInvertAnnotator, PlainMaskAugInvertVideoAnnotator, ExpandMaskVideoAnnotator
from .maskaug import MaskAugAnnotator
from .mask import MaskDrawAnnotator
from .canvas import RegionCanvasAnnotator

# --- Text Processing Annotators ---
# English: Annotators for processing and enhancing text prompts.
# Chinese: 用于处理和增强文本提示的处理器。
from .prompt_extend import PromptExtendAnnotator