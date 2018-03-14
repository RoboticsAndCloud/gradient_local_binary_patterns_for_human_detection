import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import train
import math

# Leemos los archivos binarios
persons = np.load('./txt/PData_bin.npy')
persons = np.reshape(persons, (-1, 5880))
print('Persons read')
backgrounds = np.load('./txt/BGData_bin.npy')
backgrounds = np.reshape(backgrounds, (-1, 5880))
print('Backgrounds read')
print('Entranando con los ultimos 2/3')

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

print('Starting to train')

svm = train.createSVM()
train.train(svm, trainHog, responses)
train.save(svm, 'saveData.dat')

testHog = np.append(personsTestData.flatten(), backgroundsTestData.flatten())
testHog = np.reshape(testHog, (-1, 5880))

testData = np.array(testHog, dtype=np.float32)
testResponse = svm.predict(testData)[1].ravel()

print('Porcentaje de personas correcto: ')
correctPersons = np.count_nonzero(testResponse[:testPersonsMax - testPersonsMin])/(testPersonsMax - testPersonsMin)*100
print("{:.2f}".format(correctPersons))
print('Porcentaje de personas incorrecto: ')
print("{:.2f}".format(100 - correctPersons))

print('Porcentaje de fondos correcto: ')
wrongBackgrounds = np.count_nonzero(testResponse[testPersonsMax - testPersonsMin:])/(testBackgroundsMax-testBackgroundsMin)*100
print("{:.2f}".format(100 - wrongBackgrounds))
print('Porcentaje de fondos incorrecto: ')
print("{:.2f}".format(wrongBackgrounds))