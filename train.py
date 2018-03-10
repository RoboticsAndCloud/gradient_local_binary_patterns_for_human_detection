import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb

# Machine learning

def train(svm, data, answers):
    trainData = np.array(data, dtype=np.float32)
    trainLabels = np.array(answers, dtype=np.float32)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels.astype(int))

def save(svm, name):
    svm.save(name)

def createSVM():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    return svm