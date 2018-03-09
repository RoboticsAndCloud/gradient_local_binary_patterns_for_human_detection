import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
fileName = 'test.png'
fileName2 = 'test1.png'
fileName3 = 'background.png'

def finalHistogram(img):
    total = []
    for i in range(0, 128-8, 8):
        for j in range(0, 64-8, 8):        
            a = img[i:i+15, j:j+15]
            tempTable = glpb.glbpTable(a)
            total = np.append(total, tempTable)
            
    return total


image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image2 = cv2.imread(fileName2)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# image3 = cv2.imread(fileName3)
# image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

final = finalHistogram(image)
finalNorm = np.linalg.norm(final, 2)
finalNorm = np.divide(final, finalNorm)
# final2 = finalHistogram(image2)
# final3 = finalHistogram(image3) 

plt.figure(1)
plt.bar(range(len(final)), final, color='r', align='center')
plt.figure(2)
plt.bar(range(len(finalNorm)), finalNorm, color='r', align='center')

# plt.figure(3)
# plt.bar(range(len(final3)), final3, color='r', align='center')
plt.show()