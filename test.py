import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import train

svm = cv2.ml.SVM_load('saveData.dat')

def getTotal():
    persons = np.load('./txt/PTest_bin.npy')
    backgrounds = np.load('./txt/BGTestNames_bin.npy')

    hog = np.append(persons, backgrounds)
    hog = np.reshape(hog, (-1,5880))
    return hog

# Asignamos el test data
hog = getTotal()
testData = np.array(hog, dtype=np.float32)

# Imprimos la respuesta
testResponse = svm.predict(testData)[1].ravel()
print('Respuesta de personas: ')
print(testResponse[:100])
print('Respuesta de fondos: ')
print(testResponse[100:])
print('Porcentaje de personas correcto: ')
print(np.count_nonzero(testResponse[:100]))
print('Porcentaje de personas incorrecto: ')
print(100 - np.count_nonzero(testResponse[:100]))

print('Porcentaje de fondos correcto: ')
print(100 - np.count_nonzero(testResponse[100:]))
print('Porcentaje de fondos incorrecto: ')
print(np.count_nonzero(testResponse[100:]))