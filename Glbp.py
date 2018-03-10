import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

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
            a = np.roll(a, 1)
            numberShift += 1
    
    return a, numberShift

def pixelValue(matrix, pos):
    a = np.zeros(8, np.uint8)
    a[0] = matrix[1][2]
    a[1] = matrix[0][2]
    a[2] = matrix[0][1]
    a[3] = matrix[0][0]
    a[4] = matrix[1][0]
    a[5] = matrix[2][0] 
    a[6] = matrix[2][1]
    a[7] = matrix[2][2]
    if (pos == 8):
        return int(a[0])
    if (pos == -1):
        return int(a[7])
    
    return int(a[pos])

def gradient(matrix, posIni, posFin):
    # final = pixelValue(matrix, posFin)
    # final2 = pixelValue(matrix, (posFin+1))
    # ini = pixelValue(matrix, posIni)
    # ini2 = pixelValue(matrix, (posIni-1))    
    # return round (math.sqrt((abs(final-final2))**2 + (abs(ini-ini2)**2)))
    return round(math.sqrt((abs(pixelValue(matrix, posFin) - pixelValue(matrix, (posFin+1))))**2 + (abs(pixelValue(matrix, posIni) - pixelValue(matrix, (posIni-1)))**2)))

def unshift(angle, posIni, posFin, numberShift):
    angle -= numberShift
    posFin -= numberShift
    posIni -= numberShift
    if angle<0:
        angle += 8
    if posIni<0:
        posIni += 8
    if posFin<0:
        posFin += 8
    return int(angle), int(posIni), int(posFin) 

def getWidth(array):
    return np.count_nonzero(array)

def getMiddleInitialFinal(vector):
    # indexes = np.flatnonzero(vector)
    # initial = indexes[0]
    # final = indexes[-1]
    # middle = (initial + final) / 2
    # middle = int(middle)
    # return middle, initial, final
    indexes = np.flatnonzero(vector)
    return int((indexes[0] + indexes[-1]) / 2), indexes[0], indexes[-1]


def discard(nonZeroIndexes, consec):
    return (len(nonZeroIndexes) == 0 or len(nonZeroIndexes) == 8 or len(consec) == 4)
    # if (len(nonZeroIndexes) == 0 or len(nonZeroIndexes) == 8 or len(consec) == 4):
    #     return True
    # else:
    #     return False
    


# Input: Matriz 3x3

def glbpData(matrix):
    a, numberShift = binaryVector(matrix)

    nonZeroIndexes = np.flatnonzero(a)
    cons = consecutive(nonZeroIndexes) # array de arrays con numeros consecutivos

    if(discard(nonZeroIndexes, cons)):
        return np.zeros(56, np.uint16)
    else:
        outputMatrix = np.zeros((7, 8), np.uint16)

        for i in range(0, len(cons)):
            vector = np.zeros((8), np.uint8)
            vector[cons[i]] = 1

            width = getWidth(vector)
            middle, initial, final = getMiddleInitialFinal(vector)
            angle, initial, final = unshift(middle, initial, final, numberShift)
            gradient2 = gradient(matrix, initial, final)
            outputMatrix[width - 1][angle] += gradient2
        return outputMatrix.flatten()

def glbpTable(cell):
    table = np.zeros((56), np.uint16)
    for i in range(0, 13):
        for j in range(0, 13):
            # matrix = cell[i:i+3, j:j+3]
            # tempVector = glbpData(matrix)
            # table += tempVector
            table += glbpData(cell[i:i+3, j:j+3])
    return table

def finalHistogram(img):
    total = []
    for i in range(0, 128-8, 8):
        for j in range(0, 64-8, 8):        
            tempTable = glbpTable(img[i:i+15, j:j+15])
            total = np.append(total, tempTable)
    norm = np.linalg.norm(total, 2) / 100
    return np.divide(total, norm)

def printGLBPHistogram(array, figure):
    plt.figure(figure)
    plt.bar(range(len(array)), array, color='r', align='center')