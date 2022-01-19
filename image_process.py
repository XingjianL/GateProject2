import cv2
import numpy as np
from numpy.core.shape_base import block
import image_prep, image_util
import time
import copy
# This file intends to apply the utilities provided in image_prep.py
# to process image/images in order to filter out features
# also a place for categorizing the images

class GateDetect:
    
    def __init__(self,
                img_dim = None,
                block_dim = None,
                ncluster = 5,
                color_threshold_prep = 30):
        self.img_prep = image_prep.ImagePrep()
        self.img_prep.updateDimension(img_dim=img_dim,block_dim=block_dim)
        self.img_prep.k_kmean = ncluster
        self.img_prep.dist_thres_color_diff = color_threshold_prep
        self.createBlocks()
        self.contour_flags = {"total_contours":0, "circle":[], "line":[]}
        self.color_flags = {"furthest_distance":0,"background":True}

    # create a 2d list of block objects
    def createBlocks(self):
        x,y = self.img_prep.num_seg
        temp = []
        for i in range(y):
            temp.append([image_util.Block(pos = (i,j)) for j in range(x)])
        self.block_list = np.array(temp)
    # create image from blocks' max value
    def block2ImgByMaxVal(self):
        img = np.zeros(self.block_list.shape,dtype=np.uint8)
        for i,block_row in enumerate(self.block_list):
            for j,block in enumerate(block_row):
                img[i][j] = block.max_color
        return img
    # assign color flags to img after kmean
    def kmeanColor(self,colors,color_thres = 50):
        flags = self.color_flags
        flags["furthest_distance"] = self.img_prep.colorsMaxDist(colors)
        if flags["furthest_distance"] < color_thres:
            flags["background"] = True
        else:
            flags["background"] = False
        return flags
    
    # mask the image based on flags 
    # process see maskImgBlock()
    # axis: 0-individual block, 1-row, 2-col
    def maskImg(self,img,color_thres = 50,axis=1):
        sliced_blocks = self.img_prep.slice(img)
        full_list = []
        if axis == 0:
            for i,row in enumerate(sliced_blocks):
                block_list = []
                for j,block in enumerate(row):
                    block = self.maskImgBlock(block,color_thres)
                    self.block_list[i][j].findMax(block)

                    block_list.append(block)
                full_list.append(self.img_prep.combineRow(block_list))
            full = self.img_prep.combineCol(full_list)
        elif axis == 1:
            for i,row in enumerate(sliced_blocks):
                row = self.img_prep.combineRow(row)
                row = self.maskImgBlock(row,color_thres)
                full_list.append(row)
            full = self.img_prep.combineCol(full_list)
        elif axis == 2:
            for i,col in enumerate(sliced_blocks.T):
                col = self.img_prep.combineCol(col)
                col = self.maskImgBlock(col,color_thres)
                full_list.append(col)
            full = self.img_prep.combineRow(full_list)
        return full
    # blur -> kmeans -> filter: color_label -> contours 
    def maskImgBlock(self,block,color_thres = 50):
        blur = self.img_prep.blur(block, is_list=False)
        label,center = self.img_prep.kmeans(block)
        color_flag = self.kmeanColor(center,color_thres)
        if color_flag["background"] is True:
            block = np.zeros(block.shape,dtype="uint8")
        else:
            block = self.img_prep.drawKmeans(label,center,block.shape)
            contours,hierarchy = self.img_prep.contour(block)
            block = np.full(block.shape,32*2,dtype="uint8")
            contoured = self.img_prep.drawContour(block, contours, random_color=False)
            cv2.imshow('block',contoured)
            cv2.waitKey(0)
        return block
    # slice -> blur -> k-mean -> combine
    def wholeImgProcess(self,img):
        sliced_blocks = self.img_prep.slice(img)
        rows = [i for i in range(self.img_prep.num_seg[1])]
        for r in rows:
            blurrow = self.img_prep.blur(sliced_blocks[r,:])
            k_mean_blur = self.img_prep.kmeansList(blurrow)
            #contour_k_mean_blur = img_prep.drawContourList(k_mean_blur)
            row_contour_k_mean_blur = self.img_prep.combineRow(k_mean_blur)
            rows[r] = row_contour_k_mean_blur
        full = self.img_prep.combineCol(rows)
        return full

if __name__ == '__main__':
    plotter = image_util.PlottingUtils()
    img_prep_m = image_prep.ImagePrep()
    img_process = GateDetect(block_dim = 21)
    #img_process_2 = GateDetect(block_dim = 11)

    ##############
    # Video file #
    ##############
    vid = cv2.VideoCapture('/home/xing/TesterCodes/OpenCV/GateProject/move_towards.avi')
    
    total_frames = 0
    time_used = []
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret:
            cv2.imshow('original',frame)
            HSVFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            cv2.imshow('hsv',HSVFrame)
            begin_time = time.perf_counter()
            HSVFrame1 = copy.deepcopy(HSVFrame)
            #whole_img_process = img_process.wholeImgProcess(frame)
            mask1_img = img_process.maskImg(HSVFrame,50,0)
            #mask2_img = img_process.maskImg(HSVFrame,50,1)
            #mask3_img = img_process.maskImg(HSVFrame,50,2)
            #mask2_img = img_process_2.maskImg(HSVFrame1,50,0)

            end_time = time.perf_counter()
            cv2.imshow('mask1',mask1_img)

            simple_img = img_process.block2ImgByMaxVal()
            cv2.imshow('simple_img',simple_img)

            plotter.clearfig()
            plotter.energyByAxis(mask1_img,0)
            plotter.energyByAxis(mask1_img,1)

            #cv2.imshow('mask2',mask2_img)
            #cv2.imshow('mask3',mask3_img)
            #cv2.imshow('full_process2',whole_img_process1)
            #cv2.imshow('full_process3',whole_img_process2)
            cv2.waitKey(1)
            total_frames += 1
            time_used.append(end_time-begin_time)
        else:
            break
    fps = 1/np.average(np.array(time_used))
    print(fps)
    cv2.destroyAllWindows()
    cv2.waitKey(1)