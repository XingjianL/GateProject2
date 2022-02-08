import cv2
import numpy as np
from numpy.core.shape_base import block
import image_prep, image_util
import time
import copy
# This file intends to apply the utilities provided in image_prep.py
# to process image/images in order to filter out features
# also a place for categorizing the images
DEBUG_LOG = True
CV_IMSHOW_TIME = 0 # 0 to wait for key press after image cycle

# class for all stuff related to Gate detection
class GateDetect:
    
    def __init__(self,
                img_dim = None,
                block_dim = None,
                ncluster = 5,
                color_threshold_prep = 30):
        self.img_prep = image_prep.ImagePrep()
        # update parameters
        self.img_prep.updateDimension(img_dim=img_dim,block_dim=block_dim)
        self.img_prep.k_kmean = ncluster
        self.img_prep.dist_thres_color_diff = color_threshold_prep
        # generate block array
        self.createBlocks()
        self.contour_flags = {"total_contours":0, "circle":[], "line":[], "background":True}
        self.color_flags = {"furthest_distance":0,"background":True}

    # create a 2d list of block objects, each relates to the property of an image slice
    def createBlocks(self):
        x,y = self.img_prep.num_seg
        temp = []
        for i in range(y):
            temp.append([image_util.Block(pos = (i,j)) for j in range(x)])
        self.block_list = np.array(temp)

    # create image from block array's max value
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
    def contourFlag(self,contours,contour_count = 1):
        flags = self.contour_flags
        flags["total_contours"] = len(contours)
        if flags["total_contours"] < contour_count:
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
        if DEBUG_LOG:
            print("-----new block-----")
        blur = self.img_prep.blur(block, is_list=False)
        label,center = self.img_prep.kmeans(block)
        color_flag = self.kmeanColor(center,color_thres)
        if color_flag["background"] is True:
            block = np.zeros(block.shape,dtype="uint8")
        else:
            block = self.img_prep.drawKmeans(label,center,block.shape)
            contours,hierarchy = self.img_prep.contour(block)
            block = np.full(block.shape,8,dtype="uint8")
            contoured = self.img_prep.drawContour(block, contours, random_color=False)
            cv2.imshow('block',contoured)
            cv2.waitKey(CV_IMSHOW_TIME)
        return block

    # slice -> contour
    def contourImg(self, img):
        sliced_blocks = self.img_prep.slice(img)
        full_list = []
        for i,row in enumerate(sliced_blocks):
            block_list = []
            for j,block in enumerate(row):
                contours,hierarchy = self.img_prep.contour(block)
                contour_flag = self.contourFlag(contours)
                if contour_flag["background"] is False:
                    block = np.full(block.shape,128,dtype="uint8")
                else:
                    block = np.full(block.shape,0,dtype="uint8")
                self.img_prep.drawContour(block, contours, random_color=True)
                block_list.append(block)
            full_list.append(self.img_prep.combineRow(block_list))
        full = self.img_prep.combineCol(full_list)
        return full

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

    # return gate center by using max contour values along axis
    def findByMaximum(self,img):
        total_energy = self.sumOfAxis(img, axis = 0)
        sort_index = np.argsort(total_energy)
        check_index = np.flip(sort_index[-22:])
        
        width_filter = 10
        center = -1
        saved_i = check_index[0]
        for i in check_index:
            if(abs(i-saved_i) > width_filter):
                center = (i + saved_i)/2
                break
            if(DEBUG_LOG):
                print(i, " ", total_energy[i])

        return int(center)

    def sumOfAxis(self, img, axis=0):
        first_channel = img[:,:,0]
        total_energy = np.sum(first_channel,axis=axis)
        
        return total_energy
if __name__ == '__main__':
    plotter = image_util.PlottingUtils()
    img_prep_m = image_prep.ImagePrep()
    ncsu_img_process = GateDetect(block_dim = 21)
    murky_img_process = GateDetect(block_dim = 21)

    ##############
    # Video file #
    ##############
    vid = cv2.VideoCapture('/home/xing/TesterCodes/OpenCV/GateProject/GateCcomp.mp4')
    
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

            whole_img_process = murky_img_process.wholeImgProcess(frame)
            
            # murky (competition) pool config
            mask1_img = murky_img_process.contourImg(frame)
            # NCSU pool config
            #mask1_img = img_process.maskImg(HSVFrame,50,0)

            gate_location = murky_img_process.findByMaximum(mask1_img)
            print("--------------- ", gate_location)
            end_time = time.perf_counter()
            
            vert_center = int(mask1_img.shape[1] / 2)
            hori_center = int(mask1_img.shape[0] / 2)
            cv2.circle(mask1_img,(gate_location,hori_center),5,(0,0,255))
            cv2.circle(mask1_img,(vert_center,hori_center),3,(255,0,0))
            cv2.imshow('mask1',mask1_img)

            simple_img = murky_img_process.block2ImgByMaxVal()
            cv2.imshow('simple_img',simple_img)

            plotter.clearfig()
            plotter.energyByAxis(mask1_img,0)
            plotter.energyByAxis(mask1_img,1)

            #cv2.imshow('mask2',mask2_img)
            #cv2.imshow('mask3',mask3_img)
            cv2.imshow('full_process2',whole_img_process)
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