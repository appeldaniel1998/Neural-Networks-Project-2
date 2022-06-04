import sys
from typing import List
import cv2

from Neuron import Neuron
from Point import Point


def kohonenFit(points: List[Point], neurons: List[List[Neuron]]):
    """
    Function fit the kohonen algorithm on given list of data points and neurons.
    Each of the lists contains the appropriate location (x,y).
    :param points:List of data points locations
    :param neurons:List of neurons locations
    :return:List of neurons with updated locations
    """
    learningRate = 0.7
    radius = 7

    for epoch in range(100):
        for pointInd in range(len(points)):  # For each datapoint
            currPoint = points[pointInd]
            closestNeuronInd = -1
            minDist = sys.maxsize  # Initialize the minimum distance

            for neuronIndX in range(len(neurons)):  # We will iterate over all neurons
                for neuronIndY in range(len(neurons[0])):
                    currDist = neurons[neuronIndX][neuronIndY].getPoint().distance(currPoint)  # we will compute the distance to the data point
                    # and find the smallest distance
                    if currDist < minDist:
                        minDist = currDist
                        closestNeuronInd = neuronIndX, neuronIndY

            minXNeigh = max(closestNeuronInd[0] - round(radius), 0)  # If the distance is lower than zero index, we will define it as zero
            maxXNeigh = min(closestNeuronInd[0] + round(radius), len(neurons) - 1)  # If the distance is bigger than array length, we will define it as the last index
            minYNeigh = max(closestNeuronInd[1] - round(radius), 0)  # If the distance is lower than zero index, we will define it as zero
            maxYNeigh = min(closestNeuronInd[1] + round(radius), len(neurons[0]) - 1)  # If the distance is bigger than array length, we will define it as the last index

            # Compute the gaussian kernel
            gaussKer = cv2.getGaussianKernel(round(radius) * 2 + 1, 5)
            # We will take only the positive part of the gaussian kernel (from the middle till the size of the radius)
            # (Because the negative is irrelevant for distances)
            gaussKer = gaussKer[round(radius):].ravel()
            # Normalization (dividing by the first element of gaussKer - i.e. the biggest value)
            gaussKer /= gaussKer[0]

            # For all neighbours in the range of the min & max we found beforehand
            for neighbourXInd in range(minXNeigh, maxXNeigh):
                for neighbourYInd in range(minYNeigh, maxYNeigh):
                    # move the neighbours according to normalized gaussian distribution
                    currNeuron = neurons[neighbourXInd][neighbourYInd]
                    distanceBetweenNeurons = int(round(Point(neighbourXInd, neighbourYInd).distance(
                        Point(closestNeuronInd[0], closestNeuronInd[1]))))
                    # distanceBetweenNeurons = abs(neighbourInd - closestNeuronInd)
                    diff = currPoint - currNeuron.getPoint()  # Compute the difference between the data point to the neuron
                    tempLearningRate = learningRate / (epoch + 1)
                    # Find the location change of the neuron according to the computing of difference * learningRate * gaussKer(radius)
                    # When the learning rate element is decreased each epoch because of the denominator (epoch + 1)
                    if distanceBetweenNeurons < len(gaussKer):
                        changeInPosX = diff[0] * tempLearningRate * gaussKer[distanceBetweenNeurons]
                        changeInPosY = diff[1] * tempLearningRate * gaussKer[distanceBetweenNeurons]
                    else:
                        continue
                    currNeuron.changePoint(changeInPosX, changeInPosY)  # Define the new location of the neuron according to the location change we found
        radius *= 0.8  # Decrease the radius

    return neurons
