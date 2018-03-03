import cv2
from matplotlib import pyplot as plt
import numpy as np

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

matrix = np.array([[180, 176,168], [179, 175,170],[169,174,180]])
matrixB = matrix > matrix[1][1]
matrixB = matrixB.astype(int)


a = np.zeros(8,np.int8)
numberShift = 0

a[0] = matrixB[1][2]
a[1] = matrixB[0][2]
a[2] = matrixB[0][1]
a[3] = matrixB[0][0]
a[4] = matrixB[1][0]
a[5] = matrixB[2][0]
a[6] = matrixB[2][1]
a[7] = matrixB[2][2]

if (np.count_nonzero(a) != 8):
    while(a[7] == 1):
        a = np.roll(a,1)
        numberShift += 1

nonZeroIndexes = np.flatnonzero(a)
cons = consecutive(nonZeroIndexes)

vector = np.zeros((4, 8))

for i in range(0, len(cons)):
    vector[i][cons[i]] = 1

    if(np.count_nonzero(vector[i]) % 2 != 0):
        indexes = np.flatnonzero(vector[i])
        middle = int((indexes[-1] + indexes[0] ) / 2)
        angle = middle - numberShift
        if(angle < 0):
            angle += 8
        width = np.count_nonzero(vector[i])
        print(width)
        print(angle)