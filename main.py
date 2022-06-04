import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Kohonen1D
import Kohonen2D
from Neuron import Neuron
from Point import Point


def generatePointsUniform(numOfPoints: int):
    """
    Function create UnUniform distributed data between x,y values of 0-1
    :param numOfPoints:desired numbers of data points
    :return: list of the data point
    """
    lst = []
    for i in range(numOfPoints):
        lst.append(Point(random.random(), random.random()))
    return lst


def generatePointsCircle(numOfPoints: int):
    """
    Function create distributed data as required in question number 2,when x,y values between  {<x.y> | 2<= x^2 +y^2 <= 4}
    :param numOfPoints:desired numbers of data points
    :return: list of the data point
    """
    lst = []
    for i in range(numOfPoints):
        x = random.random() *2 # take random number between 0 to 2
        y = math.sqrt(4-(x**2))
        point = Point(x,y)
        lst.append(point)
    return lst


def generateNeurons1D(numOfNeurons: int):
    """
    Function define 1D array of neuron in given size
    :param numOfNeurons:desired numbers of neurons
    :return:list of the neurons
    """
    lst = []
    for i in range(numOfNeurons):
        lst.append(Neuron(Point(random.random(), random.random())))
    return lst


def generateNeurons2D(numOfNeurons: int):
    """
    Function define 2D array of neuron in given size
    :param numOfNeurons:desired numbers of neurons
    :return:list of the neurons
    """
    lst = []
    for i in range(10):
        tempLst = []
        for j in range(int(numOfNeurons / 10)):
            tempLst.append(Neuron(Point(random.random(), random.random())))
        lst.append(tempLst)
    return lst


def main():
    numOfPoints = 100  # Define number od data points
    numOfNeurons = 100  # Define number of neurons
    # points = generatePointsUniform(numOfPoints)  # For uniform
    # points = generatePointsNonUniform(numOfPoints)  # For non-uniform
    points = generatePointsCircle(numOfPoints) # For circle
    # neurons = generateNeurons1D(numOfNeurons)  # For 1D
    neurons = generateNeurons2D(numOfNeurons)  # For 2D
    # neurons = Kohonen1D.kohonenFit(points, neurons)  # For 1D
    neurons = Kohonen2D.kohonenFit(points, neurons)  # For 2D

    # To perform scatter graph of the data points and the neurons after training, we will create DF of X and Y values of data points and neurons locations
    pointsX = np.zeros(numOfPoints)
    for i in range(numOfPoints):
        pointsX[i] = points[i].getX()

    pointsY = np.zeros(numOfPoints)
    for i in range(numOfPoints):
        pointsY[i] = points[i].getY()

    neurons = np.array(neurons).ravel()
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
