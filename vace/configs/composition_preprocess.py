# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict

#------------------------ CompositionBase ------------------------#
# English: A base configuration for a generic composition task, combining two sets of frames and masks.
# Chinese: 一个通用的基础组合任务配置，用于合并两组视频帧和蒙版。
comp_anno = EasyDict()
comp_anno.NAME = "CompositionAnnotator"
comp_anno.INPUTS = {"process_type_1": None, "process_type_2": None, "frames_1": None, "frames_2": None, "masks_1": None, "masks_2": None}
comp_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ ReferenceAnything ------------------------#
# English: Configuration for the "Reference Anything" task, which extracts a subject from reference images.
# Chinese: “任意参考”任务的配置，用于从参考图像中提取一个主体。
comp_refany_anno = EasyDict()
comp_refany_anno.NAME = "ReferenceAnythingAnnotator"
# English: Nested configuration for the subject extraction sub-task.
# Chinese: 用于主体提取子任务的嵌套配置。
comp_refany_anno.SUBJECT = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                            "INPAINTING": {"MODE": "all",
                                           "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                            "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                      "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                      "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                            "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                     "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_refany_anno.INPUTS = {"images": None, "mode": None, "mask_cfg": None}
comp_refany_anno.OUTPUTS = {"images": None}


#------------------------ AnimateAnything ------------------------#
# English: Configuration for the "Animate Anything" task, which animates a subject from a reference image using a pose video.
# Chinese: “动画化任意物体”任务的配置，使用一个姿态视频来驱动参考图中的主体，生成动画。
comp_aniany_anno = EasyDict()
comp_aniany_anno.NAME = "AnimateAnythingAnnotator"
# English: Nested configuration for the pose estimation sub-task.
# Chinese: 用于姿态估计子任务的嵌套配置。
comp_aniany_anno.POSE = {"DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
                         "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"}
comp_aniany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_aniany_anno.INPUTS = {"frames": None, "images": None, "mode": None, "mask_cfg": None}
comp_aniany_anno.OUTPUTS = {"frames": None, "images": None}


#------------------------ SwapAnything ------------------------#
# English: Configuration for the "Swap Anything" task, which swaps a subject from a reference image into a target video.
# Chinese: “替换任意物体”任务的配置，用于将参考图中的主体替换到目标视频中。
comp_swapany_anno = EasyDict()
comp_swapany_anno.NAME = "SwapAnythingAnnotator"
# English: Nested configuration for extracting the reference subject.
# Chinese: 用于提取参考主体的嵌套配置。
comp_swapany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_swapany_anno.INPAINTING = {"MODE": "all",
                                # English: Nested configuration for segmenting the object to be replaced in the target video.
                                # Chinese: 用于分割目标视频中待替换物体的嵌套配置。
                                "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                                "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                          "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                          "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                                "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                         "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}
comp_swapany_anno.INPUTS = {"frames": None, "video": None, "images": None, "mask": None, "bbox": None, "label": None, "caption": None, "mode": None, "mask_cfg": None}
comp_swapany_anno.OUTPUTS = {"frames": None, "images": None, "masks": None}



#------------------------ ExpandAnything ------------------------#
# English: Configuration for the "Expand Anything" task, combining subject extraction, outpainting, and video expansion.
# Chinese: “扩展任意物体”任务的配置，结合了主体提取、图像外扩和视频扩展。
comp_expany_anno = EasyDict()
comp_expany_anno.NAME = "ExpandAnythingAnnotator"
# English: Nested configuration for the reference subject extraction sub-task.
# Chinese: 用于参考主体提取子任务的嵌套配置。
comp_expany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_expany_anno.OUTPAINTING = {"RETURN_MASK": True, "KEEP_PADDING_RATIO": 1, "MASK_COLOR": "gray"}
# English: Nested configuration for the outpainting sub-task.
# Chinese: 用于图像外扩子任务的嵌套配置。
comp_expany_anno.FRAMEREF = {}
# English: Nested configuration for the frame expansion (video generation) sub-task.
# Chinese: 用于视频帧扩展（视频生成）子任务的嵌套配置。
comp_expany_anno.INPUTS = {"images": None, "mode": None, "mask_cfg": None, "direction": None, "expand_ratio": None, "expand_num": None}
comp_expany_anno.OUTPUTS = {"frames": None, "images": None, "masks": None}


#------------------------ MoveAnything ------------------------#
# English: Configuration for the "Move Anything" task, which creates motion for an object in a static image based on a trajectory.
# Chinese: “移动任意物体”任务的配置，根据定义的轨迹为静态图像中的物体创建运动。
comp_moveany_anno = EasyDict()
comp_moveany_anno.NAME = "MoveAnythingAnnotator"
# English: Nested configuration for the layout generation from bounding boxes sub-task.
# Chinese: 用于从边界框生成布局的子任务的嵌套配置。
comp_moveany_anno.LAYOUTBBOX = {"RAM_TAG_COLOR_PATH": "models/VACE-Annotators/layout/ram_tag_color_list.txt"}
comp_moveany_anno.INPUTS = {"image": None, "bbox": None, "label": None, "expand_num": None}
comp_moveany_anno.OUTPUTS = {"frames": None, "masks": None}
