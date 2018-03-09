import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time

def finalHistogram(img):
    total = []
    for i in range(0, 128-8, 8):
        for j in range(0, 64-8, 8):        
            tempTable = glpb.glbpTable(img[i:i+15, j:j+15])
            total = np.append(total, tempTable)
    norm = np.linalg.norm(total, 2)
    return np.divide(total, norm)

fileName = 'test.png'
fileName2 = 'test1.png'
fileName3 = 'background.png'

image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

t0 = time.time()

final = finalHistogram(image)

t1 = time.time()

print(t1-t0)

# plt.figure(1)
# plt.bar(range(len(final)), final, color='r', align='center')

# plt.show()