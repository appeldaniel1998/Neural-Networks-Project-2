import sys
from typing import List

from Point import Point
from Neuron import Neuron
import numpy as np


# https://www.youtube.com/watch?v=g8O6e9C_CfY

def nearestNeuronToPoint(neurons: List[Neuron], points: List[Point], pointInd: int):
    """
    Function find the closest neuron to given data point
    :param neurons: List of all neurons
    :param points: List of all points
    :param pointInd:
    :return:
    """
    closestNeuronInd = -1
    distToClosestNeuron = sys.maxsize  # Initialize the minimum distance
    for neuronInd in range(len(neurons)):  # Iterate over all neurons
        # Compute the distance according to Euclidian distance formula
        currDist = np.sqrt((points[pointInd].getX() - neurons[neuronInd].getWeights()[pointInd][0]) ** 2 +
                           (points[pointInd].getY() - neurons[neuronInd].getWeights()[pointInd][1]) ** 2)
        # If the distance we computed is smaller than distToClosestNeuron we will define it as the new distToClosestNeuron
        if currDist < distToClosestNeuron:
            distToClosestNeuron = currDist
            closesNeuronInd = neuronInd  # And save the index of the new closest neuron
    return closestNeuronInd  # Return the index of the closest neuron


def kohonenAlgoFit(points: List[Point], neurons: List[Neuron]):
    learningRate = 0.2
    learningRateChange = 0.2
    neighbourhoodSize = 0.2
    neighbourhoodSizeChange = 0.2
    radius = 2

    for epoch in range(5):
        # We will find the closest neuron to the chosen data point
        for pointInd in range(len(points)):
            closestNeuronInd = nearestNeuronToPoint(neurons, points, pointInd)  # Select the 'winning neuron'
            # We will find all the neighbors of the 'winning neuron' (neighbours are defined by distance smaller than radius)
            for neuronInd in range(len(neurons)):
                distanceBetweenNeurons = neurons[closestNeuronInd].getPoint().distance(neurons[neuronInd].getPoint())
                if distanceBetweenNeurons <= radius:
                    # Update the weight of all neighbors
                    currLearningRate = learningRate * np.exp(-(epoch / learningRateChange))
                    currNeighborhoodSize = neighbourhoodSize * np.exp(-(epoch / neighbourhoodSizeChange))
                    currTopologicalNeighbourhood = np.exp(-(distanceBetweenNeurons ** 2 / (2 * (currNeighborhoodSize ** 2))))

                    neuron = neurons[neuronInd]
                    weights = neuron.getWeights()
                    for i in range(2):
                        weights[pointInd][i] = currLearningRate * currTopologicalNeighbourhood * (points[pointInd].getX() - weights[pointInd][i])
                    neuron.setWeights(weights)

                    lst = []
                    for i in range(3):
                        lst.append(weights[i][0] * weights[i][1])
                    neuron.setPoint(points[:3], lst)
        radius /= 2
    return neurons



