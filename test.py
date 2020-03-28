import cv2
import sys
sys.path.insert(1, 'models/PSMNet')
from generate_disparity import generate_disparity

left = cv2.imread('left.png')
right = cv2.imread('right.png')
generate_disparity(left, right, False)
