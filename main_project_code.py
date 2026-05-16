import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/ESRGAN master')

!ls /content/drive/MyDrive/Colab Notebooks/ESRGAN master

from google.colab import files
# files.upload() # Commented for automation

import RRDBNet_arch as arch

import cv2
import numpy as np
import torch
import functools
import RRDBNet_arch as arch
from google.colab import files
import matplotlib.pyplot as plt
