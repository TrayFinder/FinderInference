"""
For project-wide constants and sharred configurations

Pattern:
  Directories -> DIR (INPUT_DIR, OUTPUT_DIR): it ends with "/", represents the folder
  Files -> FILE (CONFIG_FILE, LOG_FILE): it contains only the filename
  Full Path -> PATH (LOG_FILE_PATH, DETECTION_MODEL_PATH): it contains the entire path (DIR + FILE)
  Numeric values/limits -> Clear descriptions (MAX_RETRIES, MIN_BUFFER_SIZE)
"""

import os

# DIRS

current_dir = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, '../../../')) + os.sep
PRODUCTION_DIR = os.path.join(ROOT_DIR, 'production') + os.sep
ASSETS_DIR = os.path.join(PRODUCTION_DIR, 'assets') + os.sep
MAIN_DIR = os.path.join(PRODUCTION_DIR, 'main') + os.sep
DATA_DIR = os.path.join(ASSETS_DIR, 'data') + os.sep
IMAGES_DIR = os.path.join(ASSETS_DIR, 'images') + os.sep
LOGS_DIR = os.path.join(ASSETS_DIR, 'logs') + os.sep
MODELS_DIR = os.path.join(ASSETS_DIR, 'models') + os.sep
INDEX_DIR = os.path.join(ASSETS_DIR, 'index') + os.sep

TARGET_SIZE = (288, 288)


# FILES AND NAMES

EMBEDDING_MODEL_NAME = 'embedding_search_mobilenetv3.onnx'
PRODUCT_DETECTOR_MODEL_NAME = 'product_detector.pt'
LOG_FILE_NAME = 'inference'

# --------------------------------------------------------------------
# PATHS

H5_PATH = DATA_DIR + 'filtered_embeddings.h5'
EMBEDDING_MODEL_PATH = MODELS_DIR + EMBEDDING_MODEL_NAME
PRODUCT_DETECTOR_MODEL_PATH = MODELS_DIR + PRODUCT_DETECTOR_MODEL_NAME
