import sys
from typing import List
import cv2

from Neuron import Neuron
from Point import Point
from main import printGraph


def kohonenFit(points: List[Point], neurons: List[Neuron]):
    """
    unction fit the kohonen algorithm on given list of data points and neurons.
    Each of the lists contains the appropriate location (x,y).
    :param points: List of data points locations
    :param neurons: List of neurons locations
    :return: List of neurons with updated locations
    """
    learningRate = 0.4
    radius = 30

    printGraph(points, len(points), neurons, len(neurons))
    for epoch in range(10):
        for pointInd in range(len(points)):  # For each datapoint
            currPoint = points[pointInd]
            closestNeuronInd = -1
            minDist = sys.maxsize  # Initialize the minimum distance

            for neuronInd in range(len(neurons)):  # We will iterate over all neurons
                currDist = neurons[neuronInd].getPoint().distance(currPoint)  # we will compute the distance to the data point
                # and find the smallest distance
                if currDist < minDist:
                    minDist = currDist
                    closestNeuronInd = neuronInd

            minNeigh = max(closestNeuronInd - round(radius), 0)  # If the distance is lower than zero index, we will define it as zero
            maxNeigh = min(closestNeuronInd + round(radius), len(neurons) - 1)  # If the distance is bigger than array length, we will define it as the last index

            # Compute the gaussian kernel
            gaussKer = cv2.getGaussianKernel(round(radius) * 2 + 1, 5)
            # We will take only the positive part of the gaussian kernel (from the middle till the size of the radius)
            # (Because the negative is irrelevant for distances)
            gaussKer = gaussKer[round(radius):].ravel()
            # Normalization (dividing by the first element of gaussKer - i.e. the biggest value)
            gaussKer /= gaussKer[0]

            # For all neighbours in the range of the min & max we found beforehand
            for neighbourInd in range(minNeigh, maxNeigh + 1):
                # move the neighbours according to normalized gaussian distribution
                currNeuron = neurons[neighbourInd]
                distanceBetweenNeurons = abs(neighbourInd - closestNeuronInd)
                diff = currPoint - currNeuron.getPoint()  # Compute the difference between the data point to the neuron
                tempLearningRate = learningRate / (epoch + 1)
                # Find the location change of the neuron according to the computing of difference * learningRate * gaussKer(radius)
                # When the learning rate element is decreased each epoch because of the denominator (epoch + 1)
                changeInPosX = diff[0] * tempLearningRate * gaussKer[distanceBetweenNeurons]
                changeInPosY = diff[1] * tempLearningRate * gaussKer[distanceBetweenNeurons]
                currNeuron.changePoint(changeInPosX, changeInPosY)  # Define the new location of the neuron according to the location change we found
        radius *= 0.5  # Decrease the radius
        printGraph(points, len(points), neurons, len(neurons))

    return neurons
