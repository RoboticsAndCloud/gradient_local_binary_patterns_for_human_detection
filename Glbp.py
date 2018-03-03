import cv2
from matplotlib import pyplot as plt
import numpy as np

matrix = np.array([[180, 176,168], [179, 175,170],[169,174,170]])
matrixB = matrix > matrix[1][1]

print(np.where(170))