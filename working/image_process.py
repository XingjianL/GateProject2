import cv2
import numpy as np
import image_prep, image_util
# This file intends to apply the utilities provided in image_prep.py
# to process image/images in order to filter out features
# also a place for categorizing the images

# class for all stuff related to Gate detection
class GateDetect:
    
    def __init__(self,
                img_dim = None,
                block_dim = None,
                ncluster = 5,
                color_threshold_prep = 30):
        self.img_prep = image_prep.ImagePrep()
        # update parameters
        self.img_prep.k_kmean = ncluster
        self.img_prep.dist_thres_color_diff = color_threshold_prep
        # generate block array
        self.color_flags = {"furthest_distance":0,"background":True}

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
    def maskImg(self,img,color_thres = 50,axis=0):
        sliced_blocks = self.img_prep.slice(img)
        full_list = []
        for i,row in enumerate(sliced_blocks):
            block_list = []
            for j,block in enumerate(row):
                block = self.maskImgBlock(block,color_thres)
                block_list.append(block)
            full_list.append(self.img_prep.combineRow(block_list))
        full = self.img_prep.combineCol(full_list)
        return full
    
    # blur (optional) -> kmeans -> filter: color_label -> contours 
    def maskImgBlock(self,block,color_thres = 50):

        #blur = cv2.medianBlur(block,5)             # gate is close
        #label,center = self.img_prep.kmeans(blur)
        label,center = self.img_prep.kmeans(block)  # gate is far
        color_flag = self.kmeanColor(center,color_thres)
        if color_flag["background"] is True:
            block = np.zeros(block.shape,dtype="uint8")
        else:
            block = self.img_prep.drawKmeans(label,center,block.shape)
            contours,hierarchy = self.img_prep.contour(block)
            block = np.full(block.shape,8,dtype="uint8")
            contoured = self.img_prep.drawContour(block, contours, random_color=False)
        return block

    # return gate center by using max contour values along axis
    # axis: 1 horizontal, 0 vertical
    def findByMaximum(self, img, axis):
        total_energy = self.sumOfAxis(img, axis = axis) # sum all pixels along an axis
        sort_index = np.argsort(total_energy)        # sort the sum
        check_index = np.flip(sort_index[-22:])      # filter the sorted array
        
        width_filter = 10   # pixel gap between peaks
        center = -1
        saved_i = check_index[0]
        if axis == 0:               # 2 vertical peaks
            for i in check_index:
                if(abs(i-saved_i) > width_filter):
                    center = (i + saved_i)/2
                    break
        else:                       # 1 horizontal peak + offset
            center = check_index[0] + 20
        return int(center)

    def sumOfAxis(self, img, axis=0):
        first_channel = img[:,:,0]
        total_energy = np.sum(first_channel,axis=axis)
        return total_energy

if __name__ == '__main__':
    plotter = image_util.PlottingUtils()
    img_prep_m = image_prep.ImagePrep()
    ncsu_img_process = GateDetect(block_dim = 21)

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

            # NCSU pool config
            mask1_img = ncsu_img_process.maskImg(HSVFrame,50,0)
            
            # identified location (x, y)
            # the height location is not reliable
            gate_location = (ncsu_img_process.findByMaximum(mask1_img,0),
                            ncsu_img_process.findByMaximum(mask1_img,1))
            # image center
            vert_center = int(mask1_img.shape[1] / 2)
            hori_center = int(mask1_img.shape[0] / 2)

            # debugging, visualize results (unnecessary for robot)
            # blue circle: image center
            # green circle: gate location
            # red circle: x: gate location, y: image center
            cv2.circle(mask1_img,gate_location,5,(0,255,0))
            cv2.circle(mask1_img,(gate_location[0],hori_center),5,(0,0,255))
            cv2.circle(mask1_img,(vert_center,hori_center),3,(255,0,0))
            cv2.imshow('mask1',mask1_img)
            plotter.clearfig()
            plotter.energyByAxis(mask1_img,0)
            plotter.energyByAxis(mask1_img,1)

            cv2.waitKey(1)
        else:
            break # no more frames, exit while loop
    cv2.destroyAllWindows()
    cv2.waitKey(1)