import numpy as np
import cv2

image_seg = cv2.imread("../inputs/left.png")
image_depth = cv2.imread('../disparity.png')
gray = cv2.cvtColor(image_depth, cv2.COLOR_RGB2GRAY)

# image_seg = np.reshape(image_seg, ())
print(image_seg.shape)
print(gray.shape)
f=open("PointCloud.pcd", "a+")

for x in range(gray.shape[0]):
    for y in range(gray.shape[1]):
        color =  65536 * image_seg[x][y][0] + 256 * image_seg[x][y][1] + image_seg[x][y][2]
        f.write(str(x) + " " + str(y) + " " + str(gray[x][y]*10) + " " +str(color) +"\n" )
f.close()
