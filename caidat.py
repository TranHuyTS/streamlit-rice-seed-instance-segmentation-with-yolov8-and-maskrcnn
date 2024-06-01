from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

IMAGE = 'Image'
SOURCES_LIST = [IMAGE]
# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'ANP3-2-B1-GB(1).jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'ANP3-2-B1-GB_jpg.rf.f0e912cf79daac9fa0826ebe5399a78b.jpg'
# ML Model config
MODEL_DIR = ROOT / 'model'
YOLOV8S_SEG_MODEL = MODEL_DIR / 'yolov8s.pt'
YOLOV8M_SEG_MODEL = MODEL_DIR / 'yolov8m.pt'
#YOLOV8L_SEG_MODEL = MODEL_DIR / 'yolov8l.pt'
#MASKRCNN_MODEL = MODEL_DIR / 'mrcnn.h5'
#DEEPLAB_MODEL = MODEL_DIR / 'yolov8m.pt'