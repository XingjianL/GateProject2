import cv2
from cv2 import contourArea
import numpy as np
import matplotlib.pyplot as plt

import image_prep

class PlottingUtils:
    plt.figure()
    plt.ion()
    def closeAllPlots(self):
        plt.close('all')
    def clearfig(self):
        plt.clf()
    # x-axis: 0, y-axis: 1
    def energyByAxis(self, img, axis = 0):
        first_channel = img[:,:,0]
        total_energy = np.sum(first_channel,axis=axis)
        
        title_axis = ["by column","by row"]
        plt.plot(total_energy)
        plt.legend(title_axis)
        plt.draw()
        plt.pause(0.01)
        return total_energy

class Block:
    def __init__(self, pos):
        self.position = pos
        self.total_energy = 0
        self.max_color = 0
    
    def findEnergy(self,img):
        self.total_energy = np.sum(img)
    def findMax(self,img):
        self.max_color = np.amax(img)
        #print(self.max_color)

class ColorAssignment:
    # concept:
    # represent 3 channels of signals in the form of colors
    # channel 1 is used to represent the presence of long edges, 
    #       where contour area must be 0 or close, strength assigned based on rect dimension,angle
    #           observed by cv2.boundingRect() and contourArea()
    # channel 2 is used to represent the presence of noise
    #       signal strength is assigned based on a combination of 
    #       the # of contours and the total area of contours
    #           noise makes it difficult to filter out previous part
    #           but it could contain important info in the context of neighbor blocks
    # channel 3 is used to represent the special locations
    #       strength is decided by location of contour
    #           in theory, all contour of gate should touch the edge of block
    def generateColors(self, contours, image_shape):
        # receive all contours and the image size
        # returns corresponding colors

        # find the noise multiplier
        # loop all contours
        # assign the clean edge values to channel 1,
        # assign the noise value to channel 2 
        # assign touching bounds values to channel 3,
        # append the color
        # loop 
        # return colors
        noise = self.noiseMultiplier(contours,image_shape)
        color_list = []
        for contour in contours:
            self.contourProperty(contour)
            color = [0,noise,0]
            color[0] = int(255 * self.cleanEdgeSignal(image_shape))
            color[2] = int(255 * self.touchingBounds(contour,image_shape))
            # if an edge is desired, it is probably not noise
            # it could be noise otherwise
            if (color[0] == 0):
                color[1] = int(255 * noise)
            print(color)
            color_list.append(color)
        return color_list
    def cleanEdgeSignal(self, image_size):
        # find angle, bounding rectangle, area
        # assign whether this contour is a desired edge
        angle = abs(self.rot_rect[2]) % 90
        if (self.area < 5 and (angle < 20 or angle > 70)
            and any(size >= image_size[0] for size in self.upright_rect[2:])):
            return 1
        return 0

    def noiseMultiplier(self,contours, image_shape):
        # find # of contours, area of all contours in relation to image size
        # return sum of contoured / image size
        zero_ground = np.zeros(image_shape,dtype="uint8")
        contoured = cv2.drawContours(zero_ground,contours,-1,(1,1,1))
        sum_contour = np.sum(contoured[:,:,0])
        return sum_contour / image_shape[0]*image_shape[1]

    def touchingBounds(self, contour, image_size):
        # find if the contour touches the image edge
        # contour_box is the bounding box (x,y,w,h)
        # check if x,y == 0, check if x+w,y+h = image_size
        x,y,w,h = self.upright_rect
        if ((x == 0 or y == 0)
            or (x+w == image_size or y+h == image_size)):
            return 1
        return 0

    def contourProperty(self,contour):
        self.rot_rect = cv2.minAreaRect(contour)
        self.upright_rect = cv2.boundingRect(contour)
        self.area = cv2.contourArea(contour)
