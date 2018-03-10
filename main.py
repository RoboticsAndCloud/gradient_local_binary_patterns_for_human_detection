import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import train


def getArrays():
    # Leemos los archivos binarios
    persons = np.load('./txt/hogPE2.txt')
    print('Persons read')
    backgrounds = np.load('./txt/hogBG2.txt')
    print('Backgrounds read')

    persons2len = len(np.reshape(persons, (-1, 5880)))
    backgrounds2len = len(np.reshape(backgrounds, (-1, 5880)))
    hog = np.append(persons, backgrounds)
    persons = None
    backgrounds = None
    hog = np.reshape(hog, (-1,5880))    

    responsesPersons = np.ones(persons2len, np.uint8)
    responsesBackgrounds = np.zeros(backgrounds2len, np.uint8)
    responses = np.append(responsesPersons, responsesBackgrounds)

    responsesPersons = None
    responsesBackgrounds = None
    persons2len = None
    backgrounds2len = None
    return hog, responses


hog, responses = getArrays()

print('Starting to train')

svm = train.createSVM()
train.train(svm, hog, responses)
train.save(svm, 'saveData.dat')