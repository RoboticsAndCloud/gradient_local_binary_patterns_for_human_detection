import cv2
import numpy as np
from matplotlib import pyplot as plt

fileName = 'test.png'

image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for i in range(0, 128, 16):
    for j in range(0, 64, 16):
        a = image[i:i+2, j:j+2]