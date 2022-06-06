import math
import random

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import Kohonen1D
import Kohonen2D
from Neuron import Neuron
from Point import Point
import monkeyHandPoints


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


def generatePointsNonUniform(numOfPoints: int):
    """
    Function create UnUniform distributed data such that:
    half of the data will be uniform distributed between x,y values of 0-1,
    quarter of the data will be distributed  between x,y values of (0-1)/3
    quarter of the data will be distributed  between x,y values of (1-(0-1))/3

    :param numOfPoints:desired numbers of data points
    :return: list of the data point
    """
    lst = []
    quarter_data = numOfPoints // 4
    half_data = numOfPoints // 2
    for i in range(quarter_data):
        lst.append(Point(random.random() / 3, random.random() / 3))
    for i in range(quarter_data):
        lst.append(Point(1 - random.random() / 3, 1 - random.random() / 3))
    for i in range(half_data):
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
        alpha = 2 * math.pi * random.random()
        r = np.sqrt(2) * math.sqrt(random.random()) + np.sqrt(2)
        x = r * math.cos(alpha)
        y = r * math.sin(alpha)
        lst.append(Point(x, y))
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


def generateNeuronsCircle(numOfNeurons: int):
    """
    Function define 1D array of neuron in given size
    :param numOfNeurons:desired numbers of neurons
    :return:list of the neurons
    """
    lst = []
    for i in range(numOfNeurons):
        lst.append(Neuron(Point(random.random() * 6 - 3, random.random() * 6 - 3)))
    return lst


def generateNeurons2D(numOfNeurons: int):
    """
    Function define 2D array of neuron in given size
    :param numOfNeurons:desired numbers of neurons
    :return:list of the neurons
    """
    lst = []
    for i in range(15):
        tempLst = []
        for j in range(15):
            tempLst.append(Neuron(Point(i / 15, j / 15)))
        lst.append(tempLst)
    return lst


def printGraph(points, numOfPoints, neurons, numOfNeurons):
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


def main():
    random.seed(1)
    img_path = 'Hand Pics/handPicFull.png'
    ######################### black_list=monkeyHandPoints.createHandMatrix(img_path)
    ########################## print("black_list:",black_list)
    ########################## random_points=monkeyHandPoints.lotteryPoints(black_list,10)
    ############################ print("random_points:",random_points)

    numOfPoints = 1000  # Define number od data points
    numOfNeurons = 225  # Define number of neurons

    points = generatePointsUniform(numOfPoints)  # For uniform
    # points = generatePointsNonUniform(numOfPoints)  # For non-uniform

    # points = generatePointsCircle(numOfPoints)  # For circle
    # neurons = generateNeuronsCircle(numOfNeurons)  # For Circle

    # neurons = generateNeurons1D(numOfNeurons)  # For 1D
    # neurons = Kohonen1D.kohonenFit(points, neurons)  # For 1D

    neurons = generateNeurons2D(numOfNeurons)  # For 2D
    neurons = Kohonen2D.kohonenFit(points, neurons)  # For 2D

    # for full monkey finger
    # black_list = monkeyHandPoints.createHandMatrix(img_path)
    # points = monkeyHandPoints.lotteryPoints(black_list, numOfPoints)
    # neurons = generateNeurons2D(numOfNeurons)  # For 2D

    # printGraph(points, numOfPoints, neurons, numOfNeurons)

    neurons = Kohonen2D.kohonenFit(points, neurons)  # For 2D

    printGraph(points, numOfPoints, neurons, numOfNeurons)


if __name__ == '__main__':
    main()
