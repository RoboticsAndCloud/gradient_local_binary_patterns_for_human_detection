import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time
import os

files = os.listdir("../../Curso detección de objetos/Datasets/Pedestrians-Dataset-Dummy/Pedestrians/")

np.savetxt('out.txt', files, fmt='%s')

h = np.loadtxt('out.txt', dtype='str')

fileName = ('../../Curso detección de objetos/Datasets/Pedestrians-Dataset-Dummy/Pedestrians/test1.png')
print(fileName)
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)