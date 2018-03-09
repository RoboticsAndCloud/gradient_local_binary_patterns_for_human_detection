import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time



fileName = 'test.png'
fileName2 = 'test1.png'
fileName3 = 'background.png'

image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

t0 = time.time()
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 1)
t1 = time.time()

print(t1-t0)
plt.show()