# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import cv2
import torch
import numpy as np
from .dwpose import util
from .dwpose.wholebody import Wholebody, HWC3, resize_image
from .utils import convert_to_numpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    # English: Draws the pose skeleton (body, hands, face) onto a canvas.
    # Chinese: 将姿态骨骼（身体、手部、面部）绘制到画布上。
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # English: Draw each part of the pose based on the flags.
    # Chinese: 根据标志位绘制姿态的各个部分。
    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class PoseAnnotator:
    # English: The base class for pose estimation, using the DWPose (Wholebody) model.
    # Chinese: 姿态估计的基类，使用 DWPose (Wholebody) 模型。
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']
        # English: Set the computation device.
        # Chinese: 设置计算设备。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # English: Initialize the Wholebody pose estimation model.
        # Chinese: 初始化 Wholebody 姿态估计模型。
        self.pose_estimation = Wholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        # English: Flags to control which parts of the pose to draw.
        # Chinese: 用于控制绘制姿态哪些部分的标志位。
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        # English: Prepares the image and calls the main processing function.
        # Chinese: 准备图像并调用主处理函数。
        image = convert_to_numpy(image)
        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2])

    def process(self, ori_img, ori_shape):
        # English: Performs pose estimation and processes the results.
        # Chinese: 执行姿态估计并处理结果。
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            # English: Run the pose estimation model.
            # Chinese: 运行姿态估计模型。
            candidate, subset, det_result = self.pose_estimation(ori_img)
            nums, keys, locs = candidate.shape
            # English: Normalize keypoint coordinates.
            # Chinese: 归一化关键点坐标。
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            # English: Filter out keypoints with low confidence.
            # Chinese: 过滤掉置信度低的关键点。
            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            # English: Draw the requested pose parts onto separate maps.
            # Chinese: 将请求的姿态部分绘制到不同的图上。
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(detected_map_body[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_body"] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(detected_map_face[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose, H, W, use_body=True, use_face=True)
                detected_map_bodyface = cv2.resize(detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                                                   interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(pose, H, W, use_hand=True, use_body=True, use_face=True)
                detected_map_handbodyface = cv2.resize(detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
                                                       interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # English: Rescale detection bounding boxes to the original image size.
            # Chinese: 将检测到的边界框缩放回原始图像尺寸。
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            return ret_data, det_result


class PoseBodyFaceAnnotator(PoseAnnotator):
    # English: A specialized annotator that draws only the body and face pose.
    # Chinese: 一个专门用于仅绘制身体和面部姿态的处理器。
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, True, False
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        # English: Calls the base class forward method and returns only the combined body-face map.
        # Chinese: 调用基类的 forward 方法，并仅返回身体-面部组合图。
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_bodyface']


class PoseBodyFaceVideoAnnotator(PoseBodyFaceAnnotator):
    # English: A wrapper to apply the body-face pose annotator to each frame of a video.
    # Chinese: 一个包装器，用于将身体-面部姿态处理器应用于视频的每一帧。
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            # English: Process each frame individually.
            # Chinese: 单独处理每一帧。
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames

class PoseBodyAnnotator(PoseAnnotator):
    # English: A specialized annotator that draws only the body pose.
    # Chinese: 一个专门用于仅绘制身体姿态的处理器。
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, False, False
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        # English: Calls the base class forward method and returns only the body map.
        # Chinese: 调用基类的 forward 方法，并仅返回身体姿态图。
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_body']


class PoseBodyVideoAnnotator(PoseBodyAnnotator):
    # English: A wrapper to apply the body-only pose annotator to each frame of a video.
    # Chinese: 一个包装器，用于将仅身体姿态处理器应用于视频的每一帧。
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            # English: Process each frame individually.
            # Chinese: 单独处理每一帧。
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames