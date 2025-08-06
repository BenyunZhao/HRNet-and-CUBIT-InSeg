import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor

# Single GPU val
# CUDA_VISIBLE_DEVICES=0,1 python detect.py > logs/v11n-seg-mawan811-infer.log 2>&1 & tail -f logs/v11n-seg-mawan811-infer.log

if __name__ == '__main__':
    model = YOLO('/home/claude/Documents/GitHub/YOLO11-12/runs-CUBIT/train/11n/weights/best.pt') # select your model.pt path
    model.predict(
                  # source='/home/claude/Documents/GitHub/YOLO11-12/images/mawan/img640-no-sharp',
                  source='/home/claude/Documents/GitHub/HR-Net-and-CUBIT-InSeg/ultralytics/cfg/datasets/mawan811/test',
                  imgsz=640,
                  project='runs',
                  name='v11x',
                  save=True,
                  show=False, 
                  conf=0.5,
                  iou=0.7,
                  show_boxes=True,  
                  # agnostic_nms=False,
                  visualize=True, # visualize model features maps
                  line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  save_crop=False, # save cropped images with results
                  retina_masks=False, # save segmentation masks
                )