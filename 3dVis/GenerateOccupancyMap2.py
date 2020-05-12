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

image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

r, g, b = 0, 0, 0
lr, lg, lb = 0, 0, 0
cut_off = 200
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if image[y][x][0] == 0 and lg < cut_off :
            image[y][x][0] = r
            lg += 1
        else :
            r = image[y][x][0]
            lg = 0

        if image[y][x][1] == 0 and lr < cut_off:
            image[y][x][1] = g
            lr += 1
        else :
            g = image[y][x][1]
            lr = 0

        if image[y][x][2] == 0 and lb <  cut_off:
            image[y][x][2] = b
            lb += 1
        else :
            b = image[y][x][2]
            lb = 0



cv2.imwrite('./maps/occupancy_map0'+'.png', image)
