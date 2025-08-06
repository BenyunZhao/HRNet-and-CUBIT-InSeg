# Single GPU val
# CUDA_VISIBLE_DEVICES=0 python val_seg.py > logs/v11x.log 2>&1 & tail -f logs/v11x.log

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import model_info


if __name__ == '__main__':
    # 加载模型
    model = YOLO('/home/claude/Documents/GitHub/YOLO11-12/runs-infra/train/starnet/v11x-starnet-s4/weights/best.pt')
    metrics=model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data='/home/claude/Documents/GitHub/HR-Net-and-CUBIT-InSeg/ultralytics/cfg/datasets/mawan811.yaml',
        split='test',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=16,  # (int) 每批的图像数量（-1 为自动批处理）
        imgsz=416,  # 输入图像的大小，可以是整数或w，h
        device=0,  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=16,  # 数据加载的工作线程数（每个DDP进程）
        save_json=False,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.25,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.7,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='runs',  # 项目名称（可选）
        name='v11x',  # 实验名称，结果保存在'project/name'目录下（可选）
        max_det=300,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
        cache=True,  # (bool) True/ram、磁盘或 False。是否缓存加载数据
    )


    print(f"mAP50-95_box: {metrics.box.map}") # map50-95
    print(f"mAP50_box: {metrics.box.map50}")  # map50
    print(f"mAP75_box: {metrics.box.map75}")  # map75

    print(f"mAP50-95_seg: {metrics.seg.map}")  # map50-95
    print(f"mAP50_seg: {metrics.seg.map50}")  # map50
    print(f"mAP75_seg: {metrics.seg.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}") # FPS

