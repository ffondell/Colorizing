from numpy import random
import numpy as np
from numpy import zeros
from matplotlib import pyplot as plt
from matplotlib import image as img
from numpy import asarray
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import cv2
import sys
import operator
import random
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

repColors = np.zeros(0)

class Node:#has a row, column, and value of the rgb in that square
    def __init__(self, row, col, rgb):
        self.row = row
        self.col = col
        self.rgb = rgb#array of colors values from 0 to 255
        #self.image = image

def coloringAgent(ogImage, w):
    leftTrainingImageColor = np.copy(np.split(np.asarray(ogImage), 2, axis=1)[0])
    rightTestingImageColor = np.copy(np.split(np.asarray(ogImage), 2, axis=1)[1])

    grayScale = ImageOps.grayscale(ogImage)

    leftTrainingImageGray = np.copy(np.split(np.asarray(grayScale), 2, axis=1)[0])
    rightTestingImageGray = np.copy(np.split(np.asarray(grayScale), 2, axis=1)[1])


    leftKmean = leftTrainingImageColor.reshape((leftTrainingImageColor.shape[0]*leftTrainingImageColor.shape[1],3))
    cluster = KMeans(n_clusters=5).fit(leftKmean)
    global repColors
    repColors = cluster.cluster_centers_.astype(int)
    print(repColors)
    recoloredLeft = recolorLeft(leftTrainingImageColor)
    recoloredRight = recolorRightImproved(leftTrainingImageGray, rightTestingImageGray, recoloredLeft, rightTestingImageColor, w)
    #print(calcError(recoloredRight, recolorLeft(rightTestingImageColor)))
    #return calcError(recoloredRight, recolorLeft(rightTestingImageColor))
    return np.concatenate((recoloredLeft, recoloredRight), axis=1)


def recolorLeft(leftTrainingImageColor):
    for x in range(len(leftTrainingImageColor)):
        for y in range(len(leftTrainingImageColor[0])):
            d=np.zeros(0)
            for i in range(len(repColors)):
                d = np.append(d, rgbDistance(leftTrainingImageColor[x][y], repColors[i]))
                leftTrainingImageColor[x][y]=repColors[d.argmin()]

    return leftTrainingImageColor

def recolorRight(leftTrainingImageGray, rightTestingImageGray, recoloredLeft, rightTestingImageColor):
    out = np.copy(rightTestingImageColor)

    rightFlats = getFlatsNine(np.asarray(rightTestingImageGray))
    leftFlats = getFlatsNine(np.asarray(leftTrainingImageGray))

    i = 0
    for x in range(1,len(rightTestingImageColor)-1):

        print(x)
        for y in range(1,len(rightTestingImageColor[0])-1):
            avgDs = np.mean(abs(leftFlats-rightFlats[i]), axis=1)
            result = np.argpartition(avgDs, 6)[:6]
            maj = np.zeros(6)
            indices = np.unravel_index(result[0], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            indices = np.unravel_index(result[1], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            indices = np.unravel_index(result[2], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            indices = np.unravel_index(result[3], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            indices = np.unravel_index(result[4], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            indices = np.unravel_index(result[5], shape=(len(out), len(out[0])))
            maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]=maj[np.where(repColors==recoloredLeft[indices[0]][indices[1]])[0][0]]+1
            out[x][y]=repColors[maj.argmax()]
            plt.plot(np.mean(rightFlats[i]), maj.argmax(), "ob")
            i+=1

    plt.title("Avg grayscale value vs Label associate")
    plt.xlabel("Avg grayscale value")
    plt.ylabel("Label associated")
    plt.show()


    return out

def recolorRightImproved(leftTrainingImageGray, rightTestingImageGray, recoloredLeft, rightTestingImageColor, w):
    out = np.copy(rightTestingImageColor)
    print(rightTestingImageGray)
    rightFlats = getFlatsNine(np.asarray(rightTestingImageGray))
    leftFlats = getFlatsNine(np.asarray(leftTrainingImageGray))

    i = 0
    """
    rw = np.random.rand(9)
    gw = np.random.rand(9)
    bw = np.random.rand(9)
    """
    #bias = random.rand(0,1)
    """
    rw = np.random.rand(9)*w
    gw = np.random.rand(9)*w
    bw = np.random.rand(9)*w
    """
    rw = np.array([w,w,w,w,w,w,w,w,w])
    gw = np.array([w,w,w,w,w,w,w,w,w])
    bw = np.array([w,w,w,w,w,w,w,w,w])
    for x in range(1,len(rightTestingImageColor)-1):
        #print(x)
        for y in range(1,len(rightTestingImageColor[0])-1):
            r = np.sum(rightFlats[i]*rw).astype(int)
            g = np.sum(rightFlats[i]*gw).astype(int)
            b = np.sum(rightFlats[i]*bw).astype(int)
            d=np.zeros(0)
            rgb = [r,g,b]
            for e in range(len(repColors)):
                d = np.append(d, rgbDistance(rgb, repColors[e]))
            out[x][y]=repColors[d.argmin()]
            #plt.plot(np.mean(rightFlats[i]), d.argmin(), "ob")
            i+=1

    """
    plt.title("Avg grayscale value vs Label associate")
    plt.xlabel("Avg grayscale value")
    plt.ylabel("Label associated")
    plt.show()
    """
    print(np.mean(rw))
    print(np.mean(gw))
    print(np.mean(bw))
    return out

def getFlatsNine(array):
    i=0
    flats = np.empty(0)
    for x in range(len(array)):
        for y in range(len(array[0])):
            if(array[x:x+3,y:y+3].shape==(3,3)):
                if(i==0):
                    i=1
                    flats = array[x:x+3,y:y+3].flatten()
                else:
                    flats = np.vstack((flats, array[x:x+3,y:y+3].flatten()))
    return flats

def rgbDistance(rgb1,rgb2):
    return sqrt((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2)

def printNodes(nodes):#prints all nodes by coordinates for debug purposes
    if(isinstance(nodes, Node)):
        print("\nCoords: "+str(nodes.row)+", "+str(nodes.col))
        #print("\nCoords: "+str(nodes.row)+", "+str(nodes.col)+" RGB: "+str(nodes.rgb[0])+", "+str(nodes.rgb[1])+", "+str(nodes.rgb[1]))
    else:
        for i in range(len(nodes)):
            print("\nCoords: "+str(nodes[i].row)+", "+str(nodes[i].col)+" Score: "+str(nodes[i].rgb))

def calcError(image1, image2):
    sum = 0.0
    for x in range(len(image1)):
        for y in range(len(image1[0])):
            sum = sum + rgbDistance(image1[x][y], image2[x][y])
    return sum

def bestFit(image):

    for x in range(1,21):
        print(x)
        plt.plot(x/20, coloringAgent(image, x/20), "ob")
    plt.axis([0, 1, 0, 10000000])
    plt.title("W vs Loss")
    plt.xlabel("W")
    plt.ylabel("Loss")
    plt.show()

image = Image.open("smalltree.jpeg")
#coloringAgent(image, 0.2)


plt.imshow(coloringAgent(image, 0.2))
plt.show()
