import random

import cv2

from Point import Point


def createHandMatrix(img_path):
    """
    Function get img_path of gray image
    and return matrix with 0 and 1 values such that 255 will be in white parts of the image and 0 will be in black parts of the image
    :param img_path:img_path of gray image
    :return:list of all pixels locations that are black
    """
    black_points = []  # Define empty list of all the black points in the image
    # Read image as black and white
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # We will take the size of the image
    height, width = img.shape
    # Iterate over all pixels in image and define it as only 0 or 1 values
    for i in range(height):
        for j in range(width):
            if img[i][j] > 127:  # Define threshold for white values
                img[i][j] = 255
            else:  # If the value is smaller than 127 we will define it as black
                img[i][j] = 0
                black_points.append(Point(j / width, 1 - i / height))  # normalised the data to be values of 0-1
    return black_points


def lotteryPoints(black_list: list, lottery_num: int) -> list:
    """
    Function get list of all pixels location where the color is black and lottery points according the given number
    :param black_list:list of all black pixels location in the image
    :param lottery_num:required number of lotteries
    :return:list of the random black points that have been chosen
    """
    random_blk_pnt = []
    for i in range(lottery_num):  # lottery time as the given number
        randomNumber = random.randint(0, len(black_list))  # choose random index of the black_list
        random_blk_pnt.append(black_list[randomNumber])  # add the point in the random index to the random_blk_pnt
    return random_blk_pnt
