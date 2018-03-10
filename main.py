import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import Train as train
import time


# h = np.loadtxt('outBG.txt', dtype='str')

# out = []

# for files in h:
#     fileName = ('./images/Background/%s' % files)
#     print(fileName)
#     image = cv2.imread(fileName)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     final = glpb.finalHistogram(image)
#     out = np.append(out, final)
    
# np.savetxt('./txt/hogBG.txt', out)


# testResponse = svm.predict(testData)[1].ravel()
# print(testResponse)

persons = np.loadtxt('/txt/persons.txt', dtype='str')
backgrounds = np.loadtxt('/txt/backgrounds.txt', dtype='str')

persons = np.reshape(persons, (-1, 5880))
backgrounds = np.reshape(backgrounds, (-1, 5880))
hog = np.append(persons, backgrounds)

responsesPersons = np.ones(len(persons), np.uint8)
responsesBackgrounds = np.ones(len(backgrounds), np.uint8)
responses = np.append(responsesPersons, responsesBackgrounds)

svm = train.createSVM()
train.train(svm, hog, responses)