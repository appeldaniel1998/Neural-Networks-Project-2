import sys
from typing import List
import cv2

from Neuron import Neuron
from Point import Point


def kohonenFit(points: List[Point], neurons: List[Neuron]):
    learningRate = 0.2
    radius = 25

    for epoch in range(10):
        for pointInd in range(len(points)):
            currPoint = points[pointInd]
            closestNeuronInd = -1
            minDist = sys.maxsize

            for neuronInd in range(len(neurons)):
                currDist = neurons[neuronInd].getPoint().distance(currPoint)
                if currDist < minDist:
                    minDist = currDist
                    closestNeuronInd = neuronInd

            minNeigh = max(closestNeuronInd - round(radius), 0)
            maxNeigh = min(closestNeuronInd + round(radius), len(neurons) - 1)

            gaussKer = cv2.getGaussianKernel(round(radius) * 2 + 1, 5)
            gaussKer = gaussKer[round(radius):].ravel()
            gaussKer /= gaussKer[0]

            for neighbourInd in range(minNeigh, maxNeigh):
                # move the neighbours according to gaussian distribution (or any other)
                currNeuron = neurons[neighbourInd]
                distanceBetweenNeurons = abs(neighbourInd - closestNeuronInd)
                diff = currPoint - currNeuron.getPoint()
                changeInPosX = diff[0] * (learningRate / (epoch + 1)) * gaussKer[distanceBetweenNeurons]
                changeInPosY = diff[1] * (learningRate / (epoch + 1)) * gaussKer[distanceBetweenNeurons]
                currNeuron.changePoint(changeInPosX, changeInPosY)
        radius *= 0.5
        learningRate *= 0.5

    return neurons
