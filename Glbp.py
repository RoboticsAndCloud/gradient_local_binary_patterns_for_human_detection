import cv2
from matplotlib import pyplot as plt
import numpy as np

matrix = np.array([[180, 176,168], [179, 175,170],[169,174,170]])
matrixB = matrix > matrix[1][1]
matrixB=matrixB.astype(int)


a=np.zeros(8,np.int8)
numberShift=0
a[0]=matrixB[1][2]
a[1]=matrixB[0][2]
a[2]=matrixB[0][1]
a[3]=matrixB[0][0]
a[4]=matrixB[1][0]
a[5]=matrixB[2][0]
a[6]=matrixB[2][1]
a[7]=matrixB[2][2]
if (np.count_nonzero(a)!=8):
    while(a[7]==1):
        a=np.roll(a,1)
        numberShift+=1
print(a)
print(np.flatnonzero(a))

