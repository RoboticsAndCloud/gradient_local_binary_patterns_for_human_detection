import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time
import os

# Generamos la lista de nombres de las fotos
files = os.listdir("./images/testbg")

np.savetxt('./txt/outBG2_DataTest.txt', files, fmt='%s')

# Leemos cada archivo, hacemos su glbp y luego lo guardamos en un archivo binario
h = np.loadtxt('./txt/outBG2_DataTest.txt', dtype='str')
out = []
i = 0
for files in h:
    fileName = ('./images/testbg/%s' % files)
    i += 1
    print(i)
    image = cv2.imread(fileName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final = glpb.finalHistogram(image)
    out = np.append(out, final)
    
np.save('./txt/hogBG2_DataTest_bin', out)
