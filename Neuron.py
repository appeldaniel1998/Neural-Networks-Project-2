import math
import random
from sympy import symbols

from Point import Point
from test import *

class Neuron:
    def __init__(self, point: Point, numOfPoints: int):
        self.point = point
        self.weights = []
        for i in range(numOfPoints):
            self.weights.append([random.random(), random.random()])

    def getPoint(self):
        return self.point

    def setPoint(self, points, distances):
        self.point = findNewLocation(points, distances)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def findNewLocation(self, points, distances) -> Point:
        """
        Function get list of 3 data points and their distances (i.e. weights) from the new neuron location
        :param points:List of 3 data points
        :param distances:List of 3 new weights after updats
        :return:The new location of the neuron (x,y)
        """
        #In given new point (as variables) and 3 point and 3 distances we will create 3 equetions with 2 variables:
        # w=sqrt((x-xnew)**2 +(y-ynew)**2
        xnew, ynew = symbols('xnew ynew')
        # First equesion
        w=distances[0]**2
        x=points[0][0]
        y=points[0][1]
        first_part=(x-xnew)**2
        second_part=(y-ynew)**2
        first_equ=first_part+second_part-w

        # x_ans=[]
        # y_ans=[]
        #
        # for i in range(len(distances)):
        #     for j in range(len(points)):
        #         c=((distances[i])**2)-(points[j][0]**2)-(points[j][1]**2)+(2*points[j][0]*xnew)-(xnew**2)
        #         #0=(ynew**2)-(2*points[j][1]*ynew)-c
        #         a=1
        #         b=-(2*points[j][1])
        #         c=-c
        #         # calculate the discriminant
        #         d = (b ** 2) - (4 * a * c)
        #         # Compute two X values
        #         x1=(-b-math.sqrt(d))/(2*a)
        #         x2=(-b+math.sqrt(d))/(2*a)
        #         # Now we will find y1,y2
        #         y1=((distances[i])**2)-(points[j][0]**2)-(points[j][1]**2)+(2*points[j][0]*x1)-(x1**2)

