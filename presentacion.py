import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glbp
import time
import os

fotos = 10
svm = cv2.ml.SVM_load('saveData.dat')
# Generamos la lista de nombres de las fotos.
files = os.listdir("./images/testData")

np.savetxt('./txt/testTxt.txt', files, fmt='%s')

# Leemos cada archivo y hacemos su glbp.
h = np.loadtxt('./txt/testTxt.txt', dtype='str')
out = []
j = 1
i = 0
for files in h:
    fileName = ('./images/testData/%s' % files)
    image = cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2GRAY)
    tempHog = glbp.finalHistogram(image)
    print()
    print(files)
    print(tempHog)
    if(i == 0 or i == 5):
        glbp.printGLBPHistogram(tempHog, j, fileName)
        j += 1
    i += 1
    out = np.append(out, tempHog)

hog = np.reshape(out, (-1,5880))

out = None
testData = np.array(hog, dtype=np.float32)
testResponse = svm.predict(testData)[1].ravel()
print()
print('Personas: ')
print(testResponse[:5])
print('Fondos: ')
print(testResponse[5:])
plt.show()