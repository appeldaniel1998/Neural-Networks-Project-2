from Point import Point


class Neuron:
    def __init__(self, point: Point):
        self.point = point

    def getPoint(self):
        return self.point

    def setPoint(self, point):
        self.point = point

    def changePoint(self, valX: float, valY: float):
        self.point.x += valX
        self.point.y += valY
