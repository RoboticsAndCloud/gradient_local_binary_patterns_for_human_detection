import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import train

svm = cv2.ml.SVM_load('svm_data.dat')

def getTotal():
    persons = np.load('./txt/hogPE2_DataTest.npy')
    backgrounds = np.load('./txt/hogBG2_DataTest.npy')

    hog = np.append(persons, backgrounds)
    hog = np.reshape(hog, (-1,5880))
    return hog

# Asignamos el test data
hog = getTotal()
testData = np.array(hog, dtype=np.float32)

# Imprimos la respuesta
testResponse = svm.predict(testData)[1].ravel()
print(testResponse)
print('Porcentaje correcto: ')
print(np.count_nonzero(testResponse))
print('Porcentaje incorrecto: ')
print(100 - np.count_nonzero(testResponse))