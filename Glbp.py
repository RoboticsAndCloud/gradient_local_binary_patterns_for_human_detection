import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def gradient(x1, x2, y1, y2):
    return math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)

# Input: Matriz 3x3
# Output: Matriz 4x3 [[width, angle, gradient]],[...]]

def glbpData(matrix):
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

    nonZeroIndexes = np.flatnonzero(a)
    cons = consecutive(nonZeroIndexes) # array de arrays con numeros consecutivos

    # vector = np.zeros((4, 8))
    output = np.zeros((4, 3), np.uint8)

    for i in range(0, len(cons)):
        vector = np.zeros((8), np.uint8)
        vector[cons[i]] = 1

        width = np.count_nonzero(vector)
        if(width % 2 != 0):
            indexes = np.flatnonzero(vector)
            middle = int((indexes[-1] + indexes[0] ) / 2)
            angle = middle - numberShift
            if(angle < 0):
                angle += 8
            
            # resta componentes y al cuadrado + resta x al cuadrado al cuadrado raiz
            output[i][0] = width
            output[i][1] = angle
            output[i][2] = round(gradient(matrix[1][0], matrix[1][2], matrix[0][1], matrix[2][1]))

    # print(output)
    return output

def glbpTable(cell):
    table = np.zeros((7, 8), np.uint8)

    for i in range(0, 13):
        for j in range(0, 13):
            matrix = cell[i:i+3, j:j+3]
            # print(matrix)
            tempMatrix = glbpData(matrix)
            print(tempMatrix)
            for i in range(0, 4):
                if tempMatrix[i][0] != 0:
                    if tempMatrix[i][2] < 0:
                        print("NEGATIVO")
                    table[tempMatrix[i][0] - 1][tempMatrix[i][1]] = tempMatrix[i][2]

    return table

y = np.random.randint(100, 256, size=(16, 16))
# print(y)

glbp = glbpTable(y)
print(glbp)

histogram = cv2.calcHist([glbp], [0], None, [256], [0, 256])
plt.plot(histogram, 'c')
#plt.imshow(glbp, cmap = 'gray')
plt.show()

