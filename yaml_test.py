import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO(
        '/home/claude/Documents/GitHub/HR-Net-and-CUBIT-InSeg/ultralytics/cfg/models/11/yolo11n-HRNet.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[512, 512])
    except Exception as e:
        print(e)
        pass
    model.fuse()
