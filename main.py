import random

import numpy as np
import pandas as pd

from Point import Point
from Neuron import Neuron
from Kohonen import *
import matplotlib.pyplot as plt


def generatePoints(numOfPoints: int):
    lst = []
    for i in range(numOfPoints):
        lst.append(Point(random.random(), random.random()))
    return lst


def generateNeurons(points: List[Point], numOfPoints: int, numOfNeurons: int):
    lst = []
    for i in range(numOfNeurons):
        lst.append(Neuron(Point(0.5, random.random()), numOfPoints))
    return lst


def main():
    numOfPoints = 1000
    numOfNeurons = 100
    points = generatePoints(numOfPoints)
    neurons = generateNeurons(numOfPoints, numOfNeurons)
    neurons = kohonenAlgoFit(points, neurons)

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
