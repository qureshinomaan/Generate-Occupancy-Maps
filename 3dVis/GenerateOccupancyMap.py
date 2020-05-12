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
cv2.imwrite('./occupancy_map0'+'.png', image)
overlay = cv2.imread('./maps/occupancy_map14.png')

def resolute(image, max_res, min_res, overlay_image) :
    for resol in range(min_res, max_res) :
        res_y = resol
        res_x = resol
        fraction = 10
        for i in range(int(image.shape[0]/res_x)) :
            for j in range(int(image.shape[1]/res_y)) :
                
                seg = image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 0]
                total, count = 0, 0
                for x in range(seg.shape[0]):
                    for y in range(seg.shape[1]):
                        if seg[x][y] != 0 :
                            count += 1
                            total += seg[x][y]
                    if count > (res_x *res_y)/fraction :
                        image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 0] = total/count
                        overlay_image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 0] = total/count
                        
                        
                seg = image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 1]
                total, count = 0, 0
                for x in range(seg.shape[0]):
                    for y in range(seg.shape[1]):
                        if seg[x][y] != 0 :
                            count += 1
                            total += seg[x][y]
                    if count > (res_x *res_y)/fraction :
                        image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 1] = total/count
                        overlay_image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 1] = total/count
                        
                        
                seg = image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 2]
                total, count = 0, 0
                for x in range(seg.shape[0]):
                    for y in range(seg.shape[1]):
                        if seg[x][y] != 0 :
                            count += 1
                            total += seg[x][y]
                    if count > (res_x *res_y)/fraction :
                        image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 2] = total/count
                        overlay_image[i*res_x:(i+1)*res_x, j*res_y:(j+1)*res_y, 2] = total/count
                        
            cv2.imwrite('./maps/occupancy_map'+ str(resol) + '.png', overlay_image)


resolute(image, 101, 100, overlay)
