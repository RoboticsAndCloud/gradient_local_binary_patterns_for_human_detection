import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time
import os

files = os.listdir("./images/Background")

np.savetxt('./txt/outBG.txt', files, fmt='%s')

h = np.loadtxt('./txt/outBG.txt', dtype='str')

# print(h)
out = []

for file in h:
    fileName = ('./images/Background/%s' % file)
    print(fileName)
    image = cv2.imread(fileName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final = glpb.finalHistogram(image)
    out = np.append(out, final)
    
np.savetxt('./txt/hogBG.txt', out)
