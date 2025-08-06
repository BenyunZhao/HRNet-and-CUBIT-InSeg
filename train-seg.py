import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# Multi-GPUs training
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 train_seg.py > train_seg.py > logs/v11x.log 2>&1 & tail -f logs/v11x.log
# Single GPU training
# export CUDA_VISIBLE_DEVICES=1
# nohup python train.py > logs/hyper-yolon-seg-mawan.log 2>&1 & tail -f logs/hyper-yolon-seg-mawan.log  
# CUDA_VISIBLE_DEVICES=1 python train_seg.py > logs/v11x.log 2>&1 & tail -f logs/v11x.log


if __name__ == '__main__':
    # 加载模型
    # model = YOLOWorld('/home/claude/Documents/ultralytics/ultralytics/cfg/models/v8/yolov8n-worldv2-seg.yaml') 
    model = YOLO('yolo11m-starnet.yaml')  # 不使用预训练权重训练

    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data='/home/claude/Documents/GitHub/HR-Net-and-CUBIT-InSeg/ultralytics/cfg/datasets/mawan811.yaml',
        epochs=300,  # (int) 训练的周期数
        time=None,  # (float, optional) 训练的总时长（小时），如果设置此值，将覆盖 epochs 参数
        patience=100,  # (int) 等待无明显改善以进行早停的周期数
        batch=16,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,  # (int | list) 输入图像的尺寸，整数或宽、高列表
        save=True,  # (bool) 是否保存训练检查点和预测结果
        save_period=-1,  # (int) 每 x 个周期保存检查点（如果小于 1 则禁用）
        cache=True,  # (bool) True/ram、磁盘或 False。是否缓存加载数据
        device='0',  # (int | str | list, optional) 运行设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=8,  # (int) 数据加载的工作线程数（每个分布式训练进程）
        project='runs',  # (str, optional) 项目名称
        name='v11x',  # (str, optional) 实验名称，结果保存在 'project/name' 目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=None,  # (bool | str) 是否使用预训练模型（布尔值）或加载权重的模型路径（字符串）
        optimizer='SGD',  # (str) 优化器类型，可选值=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=0,  # (int) 用于可重复性的随机种子
        deterministic=True,  # (bool) 是否启用确定性模式
        single_cls=False,  # (bool) 将多类数据训练为单类
        rect=False,  # (bool) 如果 mode='train' 则为矩形训练，如果 mode='val' 则为矩形验证
        cos_lr=False,  # (bool) 使用余弦学习率调度器
        close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度训练，选择=[True, False]
        fraction=1.0,  # (float) 要训练的数据集的比例（默认值为1.0，即全部图像）
        profile=False,  # (bool) 在训练期间为记录器启用 ONNX 和 TensorRT 的速度分析
        freeze=None,  # (int | list, optional) 冻结前 n 层，或冻结指定层的索引列表
        multi_scale=False,  # (bool) 是否在训练期间使用多尺度

        # 验证/测试设置
        val=True,  # (bool) 在训练期间进行验证/测试
        split='val',  # (str) 用于验证的数据集划分，例如 'val', 'test' 或 'train'
        save_json=False,  # (bool) 将结果保存为 JSON 文件
        save_hybrid=False,  # (bool) 保存混合版标签（标签 + 额外预测）
        conf=0.001,  # (float, optional) 检测的对象置信度阈值（默认值为预测时 0.25，验证时 0.001）
        iou=0.7,  # (float) 非极大值抑制 (NMS) 的交并比阈值
        max_det=300,  # (int) 每张图像的最大检测数量
        half=False,  # (bool) 使用半精度（FP16）
        dnn=False,  # (bool) 使用 OpenCV DNN 进行 ONNX 推理
        plots=True,  # (bool) 在训练/验证期间保存图表和图像

        # 超参数
        lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
        momentum=0.937,  # (float) SGD 动量/Adam beta1
        weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        warmup_epochs=3.0,  # (float) 预热周期（可以使用小数）
        warmup_momentum=0.8,  # (float) 预热初始动量
        warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        box=7.5,  # (float) 盒损失增益
        cls=0.5,  # (float) 类别损失增益（与像素比例相关）
        dfl=1.5,  # (float) dfl 损失增益
        pose=12.0,  # (float) 姿势损失增益
        kobj=1.0,  # (float) 关键点对象损失增益
        label_smoothing=0.0,  # (float) 标签平滑（分数）
        nbs=64,  # (int) 名义批量大小
        hsv_h=0.015,  # (float) 图像 HSV-Hue 增强（分数）
        hsv_s=0.7,  # (float) 图像 HSV-Saturation 增强（分数）
        hsv_v=0.4,  # (float) 图像 HSV-Value 增强（分数）
        degrees=0.0,  # (float) 图像旋转（+/- 度）
        translate=0.1,  # (float) 图像平移（+/- 分数）
        scale=0.5,  # (float) 图像缩放（+/- 增益）
        shear=0.0,  # (float) 图像剪切（+/- 度）
        perspective=0.0,  # (float) 图像透视（+/- 分数），范围为 0-0.001
        flipud=0.5,  # (float) 图像上下翻转（概率）
        fliplr=0.5,  # (float) 图像左右翻转（概率）
        mosaic=1.0,  # (float) 图像马赛克（概率）
        mixup=0.5,  # (float) 图像混合（概率）
        copy_paste=0.0,  # (float) 分割的复制-粘贴（概率）
        auto_augment='randaugment',  # (str) 分类的自动增强策略
)