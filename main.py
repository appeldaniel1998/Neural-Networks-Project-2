import random
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Kohonen import *
from Neuron import Neuron
from Point import Point


def generatePoints(numOfPoints: int):
    lst = []
    for i in range(numOfPoints):
        lst.append(Point(random.random(), random.random()))
    return lst


def generateNeurons(numOfNeurons: int):
    lst = []
    for i in range(numOfNeurons):
        lst.append(Neuron(Point(random.random(), random.random())))
    return lst


def main():
    numOfPoints = 1000
    numOfNeurons = 100
    points = generatePoints(numOfPoints)
    neurons = generateNeurons(numOfNeurons)
    neurons = kohonenFit(points, neurons)

    pointsX = np.zeros(numOfPoints)
    for i in range(numOfPoints):
        pointsX[i] = points[i].getX()

    pointsY = np.zeros(numOfPoints)
    for i in range(numOfPoints):
        pointsY[i] = points[i].getY()

    neuronsX = np.zeros(numOfNeurons)
    for i in range(numOfNeurons):
        neuronsX[i] = neurons[i].getPoint().getX()

    neuronsY = np.zeros(numOfNeurons)
    for i in range(numOfNeurons):
        neuronsY[i] = neurons[i].getPoint().getY()

    dfPoints = pd.DataFrame(np.array([pointsX, pointsY]).T)
    dfNeurons = pd.DataFrame(np.array([neuronsX, neuronsY]).T)
    plt.scatter(dfPoints[0], dfPoints[1])
    plt.scatter(dfNeurons[0], dfNeurons[1])
    plt.show()


if __name__ == '__main__':
    main()
