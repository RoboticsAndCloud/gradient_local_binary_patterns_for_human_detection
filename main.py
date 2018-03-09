import cv2
import numpy as np
from matplotlib import pyplot as plt
import Glbp as glpb
import time



fileName = './images/person/person1.png'
fileName = './images/person/person2.png'
fileName = './images/person/person3.png'
fileName = './images/backgrounds/bg1.png'
fileName = './images/backgrounds/bg2.png'
fileName = './images/backgrounds/bg3.png'

fileName = './images/person/person1.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 1)

fileName = './images/person/person2.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 2)

fileName = './images/person/person3.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 3)

fileName = './images/backgrounds/bg1.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 4)

fileName = './images/backgrounds/bg2.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 5)

fileName = './images/backgrounds/bg3.png'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final = glpb.finalHistogram(image)
glpb.printGLBPHistogram(final, 6)

plt.show()