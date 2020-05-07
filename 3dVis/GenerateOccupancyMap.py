import numpy as np
import cv2

image = np.zeros((1001, 1001, 3))
file1 = open('pointCloud.pcd')
Lines = file1.readlines()

print(len(Lines))

for x in range(1000):
    for y in range(1000):
        image[x][y][0] = 0
        image[x][y][1] = 0
        image[x][y][2] = 0

for i in range(9, len(Lines)):
    pixels = [int(float(x)) for x in Lines[i].split(' ')]

    r = pixels[3]%256
    pixels[3] -= r
    g = (pixels[3]%65536)/256
    pixels[3] -= 256*g
    b = pixels[3]/65536
    if pixels[2] < 1000 and pixels[1] < 1000  and pixels[2]>0 and pixels[1]>0:
        image[1000 - pixels[1]][1000 - pixels[2]][0] = r
        image[1000 - pixels[1]][1000 - pixels[2]][1] = g
        image[1000 - pixels[1]][1000 - pixels[2]][2] = b


cv2.imwrite('./maps/occupancy_map0'+'.png', image)

actual_res = 20
cut_off = 20
color_range = 16

for res in range(4, actual_res):
    print("Res : ", res)
    for i in range(int(1000/res)):
        for j in range(int(1000/res)):

            seg = image[i*res:(i+1)*res,j*res:(j+1)*res,0]
            color_array = [0 for x in range(int(256/color_range))]
            color_array[0] = -10000
            for x in range(seg.shape[0]):
                for y in range(seg.shape[1]):
                    color_array[int(seg[x][y]/color_range)] += 1
                    
                    
            r = color_array.index(max(color_array)) * 16
            if max(color_array) > int(res*res/cut_off) :
                image[i*res:(i+1)*res,j*res:(j+1)*res,0] = int(r)
            else :
                image[i*res:(i+1)*res,j*res:(j+1)*res,0] = 0



            seg = image[i*res:(i+1)*res,j*res:(j+1)*res,1]
            color_array = [0 for x in range(int(256/color_range))]
            color_array[0] = -10000
            for x in range(seg.shape[0]):
                for y in range(seg.shape[1]):
                    color_array[int(seg[x][y]/color_range)] += 1
                    
            g = color_array.index(max(color_array)) * 16
            if max(color_array) > int(res*res/cut_off) :
                image[i*res:(i+1)*res,j*res:(j+1)*res,1] = int(g)
            else :
                image[i*res:(i+1)*res,j*res:(j+1)*res,1] = 0


            seg = image[i*res:(i+1)*res,j*res:(j+1)*res,2]
            color_array = [0 for x in range(int(256/color_range))]
            color_array[0] = -10000
            for x in range(seg.shape[0]):
                for y in range(seg.shape[1]):
                    color_array[int(seg[x][y]/color_range)] += 1
            b = color_array.index(max(color_array)) * 16
            if max(color_array) > int(res*res/cut_off):
                image[i*res:(i+1)*res,j*res:(j+1)*res,2] = int(b)
            else :
                image[i*res:(i+1)*res,j*res:(j+1)*res,2] = 0
                
                
                
    cv2.imwrite('./maps/occupancy_map'+str(res)+'.png', image)
