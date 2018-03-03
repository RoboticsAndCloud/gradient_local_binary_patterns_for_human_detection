import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def binaryVector(matrix):
    matrixB = matrix > matrix[1][1]
    matrixB = matrixB.astype(int)
    a = np.zeros(8, np.uint8)
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
    
    return a,numberShift

def pixelValue(matrix,pos):
    a = np.zeros(8, np.uint8)
    a[0] = matrix[1][2]
    a[1] = matrix[0][2]
    a[2] = matrix[0][1]
    a[3] = matrix[0][0]
    a[4] = matrix[1][0]
    a[5] = matrix[2][0] 
    a[6] = matrix[2][1]
    a[7] = matrix[2][2]
    if (pos==8):
        return int(a[0])
    if (pos==-1):
        return int(a[7])
    
    return int(a[pos])

def gradient(matrix, posIni, posFin):
    final=pixelValue(matrix,posFin)
    final2=pixelValue(matrix,(posFin+1))
    ini=pixelValue(matrix,posIni)
    ini2=pixelValue(matrix,(posIni-1))    
    return int (math.sqrt((abs(final-final2))**2+(abs(ini-ini2)**2)))

def unShift(angle,posIni,posFin,numberShift):
    angle-=numberShift
    posFin-=numberShift
    posIni-=numberShift
    if angle<0:
        angle+=8
    if posIni<0:
        posIni+=8
    if posFin<0:
        posFin+=8
    return int(angle),int(posIni),int(posFin) 




