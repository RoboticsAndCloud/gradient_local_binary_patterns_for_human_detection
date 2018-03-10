import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)


fileName = './images/person/person1.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 1)

fileName = './images/person/person2.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final2 = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 2)

# fileName = './images/person/person3.png'
# image = cv2.imread(fileName)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# final3 = glpb.finalHistogram(image)
# glpb.printGLBPHistogram(final, 3)

fileName = './images/backgrounds/bg1.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final4 = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 4)

fileName = './images/backgrounds/bg2.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final5 = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 5)

# fileName = './images/backgrounds/bg3.png'
# image = cv2.imread(fileName)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# final6 = glpb.finalHistogram(image)
# glpb.printGLBPHistogram(final, 6)

print(final)

# Machine learning

trainData = np.array([final, final2, final4], dtype=np.float32)
trainLabels = np.array([1, 1, 0], dtype=np.float32)

testData = np.array([final5], dtype=np.float32)

svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels.astype(int))
svm.save('svm_data.dat')

testResponse = svm.predict(testData)[1].ravel()
print(testResponse)
plt.show()