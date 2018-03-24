import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glbp
import train
import math

# Leemos los archivos binarios con la data de los GLBP previamente calculados.
persons = np.load('./txt/PData_bin.npy')
persons = np.reshape(persons, (-1, 5880))
print('%s fotos de personas encontradas...' % len(persons))
backgrounds = np.load('./txt/BGData_bin.npy')
backgrounds = np.reshape(backgrounds, (-1, 5880))
print('%s fotos de fondos encontradas...' % len(backgrounds))
print('Entrenaremos con los últimos 2/3 del dataset...')


trainPersonsMin = math.floor(len(persons)*1/3)
trainPersonsMax = len(persons)
testPersonsMin = 0
testPersonsMax = math.floor(len(persons)*1/3)

trainBackgroundsMin = math.floor(len(backgrounds)*1/3)
trainBackgroundsMax = len(backgrounds)
testBackgroundsMin = 0
testBackgroundsMax = math.floor(len(backgrounds)*1/3)

personsTrainData = persons[trainPersonsMin:trainPersonsMax]
personsTestData = persons[testPersonsMin:testPersonsMax]

backgroundsTrainData = backgrounds[trainBackgroundsMin:trainBackgroundsMax]
backgroundsTestData = backgrounds[testBackgroundsMin:testBackgroundsMax]

responsesPersons = np.ones(len(personsTrainData), np.uint8)
responsesBackgrounds = np.zeros(len(backgroundsTrainData), np.uint8)
responses = np.append(responsesPersons, responsesBackgrounds)

trainHog = np.append(personsTrainData.flatten(), backgroundsTrainData.flatten())
trainHog = np.reshape(trainHog, (-1,5880))

print('Entrenando...')

svm = train.createSVM()
train.train(svm, trainHog, responses)
train.save(svm, 'saveData.dat')
print('Probando...')

testHog = np.append(personsTestData.flatten(), backgroundsTestData.flatten())
testHog = np.reshape(testHog, (-1, 5880))

testData = np.array(testHog, dtype=np.float32)
testResponse = svm.predict(testData)[1].ravel()

correctPersons = np.count_nonzero(testResponse[:testPersonsMax - testPersonsMin])/(testPersonsMax - testPersonsMin)*100
print('Personas correctas: {:.2f}%'.format(correctPersons))
print('Personas incorrectas: {:.2f}%'.format(100 - correctPersons))

wrongBackgrounds = np.count_nonzero(testResponse[testPersonsMax - testPersonsMin:])/(testBackgroundsMax-testBackgroundsMin)*100
print('Fondos correctos: {:.2f}%' .format(100 - wrongBackgrounds))
print('Fondos incorrectos: {:.2f}%' .format(wrongBackgrounds))