import numpy as np
import cv2


calibration_matrix =[[9.597910e+02, 0.000000e+00, 6.960217e+02],
                    [0.000000e+00, 9.569251e+02, 2.241806e+02],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00]]
                    

                    
fx = calibration_matrix[0][0]
ox = calibration_matrix[0][2]
fy = calibration_matrix[1][1]
oy = calibration_matrix[1][2]

image_seg = cv2.imread("../outputs/instance_segmentation/IS_left.jpg")
image_depth = cv2.imread('../disparity.png')
gray = cv2.cvtColor(image_depth, cv2.COLOR_RGB2GRAY)
max_disparity = 100
# image_seg = np.reshape(image_seg, ())
print(image_seg.shape)
print(gray.shape)
f=open("PointCloud.pcd", "a+")

for x in range(gray.shape[0]):
    for y in range(gray.shape[1]):
        color =  65536 * image_seg[x][y][0] + 256 * image_seg[x][y][1] + image_seg[x][y][2]
        disparity = gray[x][y]
        Z = 721*(fx)/(disparity)
        X = 721*(1242 - x - ox)/disparity
        Y = 721*(y - oy)/disparity
        
        f.write(str(X) + " " + str(Y) + " " + str(Z) + " " +str(color) +"\n" )
f.close()
