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
    if (r == 220 and b == 220 and g == 220) == False :
        if pixels[2] < 1000 and pixels[1] < 1000  and pixels[2]>0 and pixels[1]>0:
            image[1000 - pixels[1]][1000 - pixels[2]][0] = r
            image[1000 - pixels[1]][1000 - pixels[2]][1] = g
            image[1000 - pixels[1]][1000 - pixels[2]][2] = b

image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite('occupancy_map0'+'.png', image)

actual_res = 15
cut_off = 8

for res in range(5, actual_res):
 for i in range(int(1000/res)):
     for j in range(int(1000/res)):

         seg = image[i*res:(i+1)*res,j*res:(j+1)*res,0]
         count = 0
         total = 0

         for x in range(seg.shape[0]):
             for y in range(seg.shape[1]):
                 if seg[x][y] != 0 :
                     count += 1
                 total += seg[x][y]

         if count != 0 :
             r = total/count
             if count > int(res*res/cut_off) :
                 image[i*res:(i+1)*res,j*res:(j+1)*res,0] = int(r)
             else :
                 image[i*res:(i+1)*res,j*res:(j+1)*res,0] = 0



         seg = image[i*res:(i+1)*res,j*res:(j+1)*res,1]
         count = 0
         total = 0

         for x in range(seg.shape[0]):
             for y in range(seg.shape[1]):
                 if seg[x][y] != 0 :
                     count += 1
                 total += seg[x][y]
         if count != 0 :
             g = total/count
             if count > int(res*res/cut_off) :
                 image[i*res:(i+1)*res,j*res:(j+1)*res,1] = int(g)
             else :
                 image[i*res:(i+1)*res,j*res:(j+1)*res,1] = 0


         seg = image[i*res:(i+1)*res,j*res:(j+1)*res,2]
         count = 0
         total = 0

         for x in range(seg.shape[0]):
             for y in range(seg.shape[1]):
                 if seg[x][y] != 0 :
                     count += 1
                 total += seg[x][y]
         if count != 0 :
             b = total/count
             if count > int(res*res/cut_off):
                 image[i*res:(i+1)*res,j*res:(j+1)*res,2] = int(b)
             else :
                 image[i*res:(i+1)*res,j*res:(j+1)*res,2] = 0
 cv2.imwrite('./maps/occupancy_map'+str(res)+'.png', image)
