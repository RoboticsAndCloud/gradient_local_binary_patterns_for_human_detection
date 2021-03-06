import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glbp
import os

# Generamos la lista de nombres de las fotos
files = os.listdir("./images/testData")

np.savetxt('./txt/testTxt.txt', files, fmt='%s')

# Leemos cada archivo, hacemos su glbp y luego lo guardamos en un archivo binario
h = np.loadtxt('./txt/testTxt.txt', dtype='str')
out = []
i = 0
for files in h:
    fileName = ('./images/testData/%s' % files)
    i += 1
    print(i)
    image = cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2GRAY)
    out = np.append(out, glbp.finalHistogram(image))
    
np.save('./txt/PTest_bin', out)
