import numpy as np
import cv2
import math

calibration_matrix =[[9.597910e+02, 0.000000e+00, 6.960217e+02],
    [0.000000e+00, 9.569251e+02, 2.241806e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]]

R = [[9.999758e-01, -5.267463e-03, -4.552439e-03],
    [5.251945e-03, 9.999804e-01, -3.413835e-03],
    [4.570332e-03, 3.389843e-03, 9.999838e-01]]
    
T = [5.956621e-02, 2.900141e-04, 2.577209e-03 ]

R_T = [[9.999758e-01, -5.267463e-03, -4.552439e-03, 5.956621e-02,],
    [5.251945e-03, 9.999804e-01, -3.413835e-03, 2.900141e-04],
    [4.570332e-03, 3.389843e-03, 9.999838e-01,  2.577209e-03]]
                    
fx = calibration_matrix[0][0]
ox = calibration_matrix[0][2]
fy = calibration_matrix[1][1]
oy = calibration_matrix[1][2]


pcd_essentials = ['# .PCD v.7 - Point Cloud Data file format',
                'VERSION .7',
                'FIELDS x y z rgb',
                'TYPE F F F I',
                'COUNT 1 1 1 1',
                'WIDTH 1242',
                'HEIGHT 375',
                'POINTS 465750',
                'DATA ascii'
]

def generate3dPC(image_disparity, image_segment) :
    image_seg = cv2.imread(image_segment)
    image_depth = cv2.imread(image_disparity)
    gray = cv2.cvtColor(image_depth, cv2.COLOR_RGB2GRAY)
    
    open('pointCloud.pcd', 'w').close()
    f = open('pointCloud.pcd', 'a+')
    for i in pcd_essentials :
        f.write(i+"\n")

    for x in range(1000):
        f.write(str(x) + " " + str(0) + " " + str(0) + " " +str(255) +"\n" )
        f.write(str(0) + " " + str(x) + " " + str(0) + " " +str(65280) +"\n" )
        f.write(str(0) + " " + str(0) + " " + str(x) + " " +str(16711680) +"\n" )


    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            color =  65536 * image_seg[x][y][0] + 256 * image_seg[x][y][1] + image_seg[x][y][2]
            if image_seg[x][y][0] == image_seg[x][y][1] and image_seg[x][y][1] == image_seg[x][y][2] :
                color = 65536*220 + 256*220 +220
            disparity = (gray[x][y]/256)
            
            Z = 0.54*(fx*0.001)/(disparity)*100
            X = 0.54*((375 - x)/721)/(disparity)*100 + 50
            Y = 0.54*((y - 621)/721)/((disparity))*100 + 600
            pi = math.pi
            angle = 11
            X1 = math.cos(angle*pi/180) * X - math.sin(angle*pi/180) * Z
            Z1 = math.sin(angle*pi/180) * X + math.cos(angle*pi/180) * Z
            X = X1
            Z = Z1
            
            if Z <700 :
                f.write(str(X) + " " + str(Y) + " " + str(Z) + " " +str(color) +"\n" )
    f.close()
    
generate3dPC('../disparity.png', '../outputs/instance_segmentation/IS_left.jpg')
