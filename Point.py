import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distance(self, other) -> float:
        return np.sqrt((self.getX() - other.getX()) ** 2 + (self.getY() - other.getY()) ** 2)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y
