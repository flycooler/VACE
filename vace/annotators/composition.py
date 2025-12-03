# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

class CompositionAnnotator:
    # English: A generic annotator to compose two sets of processed frames and masks based on their process types.
    # Chinese: 一个通用的处理器，用于根据两个处理流程的类型，将它们各自生成的视频帧和蒙版进行合成。
    def __init__(self, cfg):
        # English: Defines the high-level categories for different processing tasks.
        # Chinese: 定义不同处理任务的高级类别。
        self.process_types = ["repaint", "extension", "control"]
        # English: Maps specific task names to their high-level categories.
        # Chinese: 将具体的任务名称映射到其高级类别。
        self.process_map = {
            "repaint": "repaint",
            "extension": "extension",
            "control": "control",
            "inpainting": "repaint",
            "outpainting": "repaint",
            "frameref": "extension",
            "clipref": "extension",
            "depth": "control",
            "flow": "control",
            "gray": "control",
            "pose": "control",
            "scribble": "control",
            "layout": "control"
        }

    def forward(self, process_type_1, process_type_2, frames_1, frames_2, masks_1, masks_2):
        # English: Merges two sets of frames and masks based on the combination of their process types.
        # Chinese: 根据处理类型的组合，合并两组视频帧和蒙版。
        total_frames = min(len(frames_1), len(frames_2), len(masks_1), len(masks_2))
        combine_type = (self.process_map[process_type_1], self.process_map[process_type_2])
        if combine_type in [("extension", "repaint"), ("extension", "control"), ("extension", "extension")]:
            output_video = [frames_2[i] * masks_1[i] + frames_1[i] * (1 - masks_1[i]) for i in range(total_frames)]
            output_mask = [masks_1[i] * masks_2[i] * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "extension"), ("control", "extension"), ("repaint", "repaint")]:
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "control"), ("control", "repaint")]:
            if combine_type in [("control", "repaint")]:
                frames_1, frames_2, masks_1, masks_2 = frames_2, frames_1, masks_2, masks_1
            output_video = [frames_1[i] * (1 - masks_1[i]) + frames_2[i] * masks_1[i] for i in range(total_frames)]
            output_mask = [masks_1[i] * 255 for i in range(total_frames)]
        elif combine_type in [("control", "control")]:  # apply masks_2
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        else:
            raise Exception("Unknown combine type")
        return output_video, output_mask


class ReferenceAnythingAnnotator:
    # English: An annotator to extract a subject from one or more reference images.
    # Chinese: 一个用于从一张或多张参考图像中提取主体的处理器。
    def __init__(self, cfg):
        from .subject import SubjectAnnotator
        # English: Internally uses a SubjectAnnotator instance to perform the actual extraction.
        # Chinese: 内部使用一个 SubjectAnnotator 实例来执行实际的提取工作。
        self.sbjref_ins = SubjectAnnotator(cfg['SUBJECT'] if 'SUBJECT' in cfg else cfg)
        self.key_map = {
            "image": "images",
            "mask": "masks"
        }
    def forward(self, images, mode=None, return_mask=None, mask_cfg=None):
        # English: Iterates through a list of images, applies subject extraction to each, and aggregates the results.
        # Chinese: 遍历一个图像列表，对每张图应用主体提取，并聚合结果。
        ret_data = {}
        for image in images:
            ret_one_data = self.sbjref_ins.forward(image=image, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
            if isinstance(ret_one_data, dict):
                for key, val in ret_one_data.items():
                    if key in self.key_map:
                        new_key = self.key_map[key]
                    else:
                        continue
                    if new_key in ret_data:
                        ret_data[new_key].append(val)
                    else:
                        ret_data[new_key] = [val]
            else:
                if 'images' in ret_data:
                    ret_data['images'].append(ret_data)
                else:
                    ret_data['images'] = [ret_data]
        return ret_data


class AnimateAnythingAnnotator:
    # English: An annotator for the "Animate Anything" workflow. It combines pose estimation and subject extraction.
    # Chinese: “动画化万物”工作流的处理器。它组合了姿态估计和主体提取。
    def __init__(self, cfg):
        from .pose import PoseBodyFaceVideoAnnotator
        # English: Initializes a pose estimator for the driving video.
        # Chinese: 初始化用于驱动视频的姿态估计器。
        self.pose_ins = PoseBodyFaceVideoAnnotator(cfg['POSE'])
        # English: Initializes a reference extractor for the subject image.
        # Chinese: 初始化用于主体图像的参考提取器。
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, frames=None, images=None, mode=None, return_mask=None, mask_cfg=None):
        # English: Extracts pose from the video and the subject from the reference images, then returns them combined.
        # Chinese: 从视频中提取姿态，从参考图像中提取主体，然后将它们组合返回。
        ret_data = {}
        ret_pose_data = self.pose_ins.forward(frames=frames)
        ret_data.update({"frames": ret_pose_data})

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class SwapAnythingAnnotator:
    # English: An annotator for the "Swap Anything" workflow. It combines video inpainting and subject extraction.
    # Chinese: “替换万物”工作流的处理器。它组合了视频修复和主体提取。
    def __init__(self, cfg):
        from .inpainting import InpaintingVideoAnnotator
        # English: Initializes an inpainting annotator to find the region to be replaced in the target video.
        # Chinese: 初始化一个视频修复处理器，用于在目标视频中找到要被替换的区域。
        self.inp_ins = InpaintingVideoAnnotator(cfg['INPAINTING'])
        # English: Initializes a reference extractor for the subject to be swapped in.
        # Chinese: 初始化一个参考提取器，用于获取要换上去的主体。
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, video=None, frames=None, images=None, mode=None, mask=None, bbox=None, label=None, caption=None, return_mask=None, mask_cfg=None):
        # English: Generates a mask for the target video and extracts the subject from reference images.
        # Chinese: 为目标视频生成蒙版，并从参考图像中提取主体。
        ret_data = {}
        mode = mode.split(',') if ',' in mode else [mode, mode]

        ret_inp_data = self.inp_ins.forward(video=video, frames=frames, mode=mode[0], mask=mask, bbox=bbox, label=label, caption=caption, mask_cfg=mask_cfg)
        ret_data.update(ret_inp_data)

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class ExpandAnythingAnnotator:
    # English: An annotator for the "Expand Anything" workflow, combining outpainting, frame expansion, and subject extraction.
    # Chinese: “扩展万物”工作流的处理器，组合了图像外扩、视频帧扩展和主体提取。
    def __init__(self, cfg):
        from .outpainting import OutpaintingAnnotator
        from .frameref import FrameRefExpandAnnotator
        # English: Initializes a reference extractor for the subject.
        # Chinese: 初始化用于主体的参考提取器。
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])
        # English: Initializes a frame expander to generate a video sequence.
        # Chinese: 初始化一个视频帧扩展器以生成视频序列。
        self.frameref_ins = FrameRefExpandAnnotator(cfg['FRAMEREF'])
        # English: Initializes an outpainting annotator to expand the scene.
        # Chinese: 初始化一个图像外扩处理器以扩展场景。
        self.outpainting_ins = OutpaintingAnnotator(cfg['OUTPAINTING'])

    def forward(self, images=None, mode=None, return_mask=None, mask_cfg=None, direction=None, expand_ratio=None, expand_num=None):
        # English: Expands the scene, generates a video from it, and extracts the reference subject.
        # Chinese: 扩展场景，从中生成视频，并提取参考主体。
        ret_data = {}
        expand_image, reference_image= images[0], images[1:]
        mode = mode.split(',') if ',' in mode else ['firstframe', mode]

        outpainting_data = self.outpainting_ins.forward(expand_image,expand_ratio=expand_ratio, direction=direction)
        outpainting_image, outpainting_mask = outpainting_data['image'], outpainting_data['mask']

        frameref_data = self.frameref_ins.forward(outpainting_image,  mode=mode[0], expand_num=expand_num)
        frames, masks = frameref_data['frames'], frameref_data['masks']
        masks[0] = outpainting_mask
        ret_data.update({"frames": frames, "masks": masks})

        ret_ref_data = self.ref_ins.forward(images=reference_image, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class MoveAnythingAnnotator:
    # English: An annotator for the "Move Anything" workflow, creating motion from a static image based on a trajectory.
    # Chinese: “移动万物”工作流的处理器，根据轨迹为静态图像创建运动。
    def __init__(self, cfg):
        from .layout import LayoutBboxAnnotator
        # English: Initializes a layout generator that creates a motion video from bounding boxes.
        # Chinese: 初始化一个布局生成器，它从边界框创建运动视频。
        self.layout_bbox_ins = LayoutBboxAnnotator(cfg['LAYOUTBBOX'])

    def forward(self, image=None, bbox=None, label=None, expand_num=None):
        # English: Generates a layout video representing the motion path.
        # Chinese: 生成代表运动路径的布局视频。
        frame_size = image.shape[:2]   # [H, W]
        ret_layout_data = self.layout_bbox_ins.forward(bbox, frame_size=frame_size, num_frames=expand_num, label=label)

        # English: Combines the original image (as the first frame) with the layout video and creates corresponding masks.
        # Chinese: 将原始图像（作为第一帧）与布局视频结合，并创建相应的蒙版。
        out_frames = [image] + ret_layout_data
        out_mask = [np.zeros(frame_size, dtype=np.uint8)] + [np.ones(frame_size, dtype=np.uint8) * 255] * len(ret_layout_data)

        ret_data = {
            "frames": out_frames,
            "masks": out_mask
        }
        return ret_data