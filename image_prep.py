from pickle import TRUE
import cv2
import numpy as np
import random
import time

# This file provides some utilities needed to process an image
# before labelling the features
DEBUG_LOG = True
class ImagePrep:
    ### Below are parameters that may be necessary to change 
    ### to suit for creating images best for labelling
    
    # array for the # of blocks in row and col
    def updateDimension(self, img_dim = None, block_dim = None):
        if img_dim is not None:
            self.img_dim_slice = img_dim
        if block_dim is not None:
            self.block_dim_slice = block_dim
        self.num_seg = np.array([ int(self.img_dim_slice[0] / self.block_dim_slice),
                                        int(self.img_dim_slice[1] / self.block_dim_slice)]) 
    
    # kmean - parameters to run cv2.kmeans()
    k_kmean = 5
    criteria_kmean = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flag_kmean = cv2.KMEANS_PP_CENTERS
    filter_kmean = True

    # canny - parameters for cv2.Canny()
    t1_canny = 7
    t2_canny = 21
    aperture_canny = 3
    L2gradient_canny = True

    # blur - parameters for blur
    bilateral_blur = 5                # higher/lower depending on visibility
    
    # colors - parameters for color manipulation
    hue_thres_color_diff = 5        # may be necessary since hue difference is very small in water
    dist_thres_color_diff = 20.5    # 3d space distance for colors, higher/lower depending on visibility

    def __init__(self):
        # slice - parameter defines how image should be sliced
        self.img_dim_slice = [210,210] # approximate dimension for image
        self.block_dim_slice = 21 # dimension for each square slices
        self.updateDimension()

    # slice the image into segments each of square blocks
    # automatically adjust the input image dimension  
    # returns a 2d matrix of image blocks, row x col
    def slice(self,img):
        seg_dim = self.num_seg
        block_dim = self.block_dim_slice
        # adjust image dimension
        if (seg_dim is not None):
            resize_dim = seg_dim * block_dim
            img = cv2.resize(img, resize_dim)
        # prepare 2d matrix to hold images
        blocks = np.empty([seg_dim[1], seg_dim[0]], dtype=np.ndarray)
        # extract blocks
        for i in range(seg_dim[0]):
            x1 = i * block_dim
            x2 = x1 + block_dim
            for j in range(seg_dim[1]):
                y1 = j * block_dim
                y2 = y1 + block_dim
                block = img[y1:y2, x1:x2]
                blocks[j][i] = block
        return blocks
    # combine an numpy array of imgs into 1 row, or 1 column
    def combineRow(self,imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=1)
        return combined_img
    def combineCol(self,imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=0)
        return combined_img
    def insertBlock(self,dst,src,block_coord):
        y_offset = block_coord[0]*src.shape[0]
        x_offset = block_coord[1]*src.shape[1]
        dst[y_offset:y_offset+src.shape[0],x_offset:x_offset+src.shape[1]] = src
        return dst

    # k-means, the same one in the OpenCV documentation
    # default parameters limit the image into 3 colors
    # works better with smaller images

    # return the label(pixels) and corresponding colors(center)
    def kmeans(self,img,k = k_kmean,criteria = criteria_kmean,flag = flag_kmean,filter = filter_kmean):
        img_kmean = img.reshape(-1,3)
        img_kmean = np.float32(img_kmean)
        ret,label,center = cv2.kmeans(img_kmean,k,None,criteria,4,flag)
        if filter is True:
            label,center = self.combineColors(center,label)
        return label,center
    def drawKmeans(self,label,center,imgShape):
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((imgShape))
        return res2
    def kmeansList(self,imgs,k = k_kmean,criteria = criteria_kmean,flag = flag_kmean):
        for i,img in enumerate(imgs):
            label, center = self.kmeans(img,k,criteria,flag)
            img = self.drawKmeans(label,center,img.shape)
            imgs[i] = img
        return imgs

    # modify the labels by combining similar colors using a color list
    # if color 1 and 2 are similar, then all pixels labeled 2 becomes 1
    def combineColors(self, colors, labels):
        labels = labels.T
        for i,colori in enumerate(colors):
            for j,colorj in enumerate(colors[i+1:]):
                j+=1
                if self.compareColorDiff(np.vstack((colori,colorj))) is False:
                    labels[0][labels[0]==j] = i # replace all j with i in labels
                    colors[j] = colors[i]
                if len(np.unique(labels[0])) == 1:
                    break
        labels = labels.T
        return labels,colors

    # return number of similar colors 
    def checkColorDiff(self,colors):
        num = 0
        for i,colori in enumerate(colors):
            for j in colors[i+1:]:
                if self.compareColorDiff(np.vstack((colori,j))) is False:
                    num += 1
                    break
        return num

    # true if colors are different, default compare the distance between colors
    # all_values = False -> only compare the first value in color space, ie. hue for HSV
    def compareColorDiff(self, colors, hue_threshold = hue_thres_color_diff, dist_threshold = dist_thres_color_diff, all_values = True):
        if all_values is True:
            dist = np.linalg.norm(colors[0]-colors[1])
            if dist > dist_threshold:
                return True
        else:
            diff = np.max(colors[:,0])-np.min(colors[:,0])
            if diff > hue_threshold:
                return True
        return False
    
    # return the biggest diameter between two colors
    def colorsMaxDist(self,colors, skip = False):
        max_dist = 0
        for i,c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                dist = np.linalg.norm(c2-c1)
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    # just a blur 
    def blur(self, imgs, region = bilateral_blur, is_list=True):
        if is_list is True:
            for i,img in enumerate(imgs):
                img = cv2.medianBlur(img,5)
                #img = cv2.GaussianBlur(img,(5,5),0)
                #img = cv2.bilateralFilter(img,region,100,100)
                imgs[i] = img
        else:
            return cv2.medianBlur(imgs,5)
            #return cv2.bilateralFilter(imgs,region,100,100)
        return imgs

    # finding contours using cv2.Canny()
    def contour(self, img):
        img_edge = cv2.Canny(img, self.t1_canny, self.t2_canny,
                                apertureSize=self.aperture_canny, L2gradient=self.L2gradient_canny)

        contours, hierarchy = cv2.findContours(img_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours,hierarchy
    def contourList(self, imgs):
        contour_list = []
        hierarchy_list = []
        for i,img in enumerate(imgs):
            contour,hierarchy = self.contour(img)
            contour_list.append(contour)
            hierarchy_list.append(hierarchy)
        return contour_list, hierarchy_list

    # create the images
    def drawContour(self, img, contours, random_color = True):
        if random_color is True:
            for i in range(len(contours)):
                img = cv2.drawContours(img,contours,i,(255*random.random(),0,255*random.random()),1)
        else:
            for i in range(len(contours)):
                color = self.contourColor(contours[i])
                img = cv2.drawContours(img,contours,i,color,1)
        return img
    def drawContourList(self, imgs):
        for i,img in enumerate(imgs):
            contours,_ = self.contour(img)
            img = self.drawContour(img, contours)
            imgs[i] = img
        return imgs

    # return the rotated_rectangle, upright_rectangle and area of the contour
    def contourProperty(self,contour):
        rot_rect = cv2.minAreaRect(contour)
        upright_rect = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        return [rot_rect, upright_rect, area]

    # determine a custom color for filtered contours
    def contourColor(self,contour):
        [rot_rect, up_rect, area] = self.contourProperty(contour)
        angle = abs(rot_rect[2]) % 90 
        if DEBUG_LOG:
            print("up_rect (x,y,w,h): ", up_rect, "area: ", area)
        
        gray_scale = 32*(1+area)
        color = (gray_scale,gray_scale,gray_scale) # black
        # filters
        if ( # check dimension of the bounding box based on block dimension
             any(size >= self.block_dim_slice / 1 for size in up_rect[2:])  
             and area < 5 
             and (angle < 20 or angle > 70) ):
            color = (255,255,255) # white
        print(color)
        return color

if __name__ == '__main__':
    timeused = []
    totalframes = 0
    img_prep = ImagePrep()
    vid = cv2.VideoCapture('/home/xing/TesterCodes/OpenCV/GateProject/move_towards.avi')
    
    while(vid.isOpened()):
        print("-------------------NEW FRAME----------------------")
        b_process = time.perf_counter()
        ret, frame = vid.read()
        if ret:
            cv2.imshow('original',frame)
            b_sb = time.perf_counter()
            sliced_blocks = img_prep.slice(frame)
            e_sb = time.perf_counter()
            t_sb = e_sb - b_sb
            print(f'slice: {t_sb:.3}')
            rows = [i for i in range(img_prep.num_seg[1])]
            for r in rows:
                b_blur = time.perf_counter()
                blurrow = img_prep.blur(sliced_blocks[r,:])
                e_blur = time.perf_counter()
                t_blur = e_blur - b_blur
                
                b_km = time.perf_counter()
                k_mean_blur = img_prep.kmeansList(blurrow)
                e_km = time.perf_counter()
                t_km = e_km - b_km
                #copy_k_mean_blur = copy.deepcopy(k_mean_blur)
                
                b_contour = time.perf_counter()
                contour_k_mean_blur = img_prep.drawContourList(k_mean_blur)
                e_contour = time.perf_counter()
                t_contour = e_contour-b_contour
                #for b,bc in enumerate(contour_k_mean_blur):
                #    cv2.imshow('block',copy_k_mean_blur[b])
                #    cv2.imshow('block_c',bc)
                #    cv2.waitKey(0)
                
                b_combine = time.perf_counter()
                row_contour_k_mean_blur = img_prep.combineRow(contour_k_mean_blur)
                rows[r] = row_contour_k_mean_blur
                e_combine = time.perf_counter()
                t_combine = e_combine-b_combine
                
                t_row = t_blur+t_km+t_contour+t_combine
                print(f"row: {r}\nblur: {t_blur:.3} kmean: {t_km:.3} contour: {t_contour:.3} combine: {t_combine:.3} total: {t_row:.3}")
            full = img_prep.combineCol(rows)
            
            e_process = time.perf_counter()
            t_process = e_process-b_process
            print(f'total: {t_process:.3}')
            
            cv2.imshow('full', full)
            cv2.waitKey(10)
            
            totalframes+=1
            timeused.append(t_process)
        else:
            break
    # average fps through out the video 
    fps = 1/np.average(np.array(timeused))
    print(f'fps: {fps}')
    cv2.destroyAllWindows()
    cv2.waitKey(1)