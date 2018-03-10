import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# Machine learning

# trainData = np.array([final, final2, final3, final7, final7, final7], dtype=np.float32)
# trainLabels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)

# testData = np.array([final, final7], dtype=np.float32)

h = np.loadtxt('outBG.txt', dtype='str')
out = []

for files in h:
    fileName = ('./images/Background/%s' % files)
    print(fileName)
    image = cv2.imread(fileName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final = glpb.finalHistogram(image)
    out = np.append(out, final)
    
np.savetxt('./txt/hogBG.txt', out)


# svm.save('svm_data.dat')
# testResponse = svm.predict(testData)[1].ravel()
# print(testResponse)
# plt.show()