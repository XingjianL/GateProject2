###
# Starting part
###
# Video paths                                                           blur
#/home/xing/TesterCodes/OpenCV/GateProject/ff.avi                       5
#/home/xing/TesterCodes/OpenCV/GateProject/gateCube1.avi                5
#/home/xing/TesterCodes/OpenCV/GateProject/gateCube2.avi                5
#/home/xing/TesterCodes/OpenCV/GateProject/gate2.avi                    5
#/home/xing/TesterCodes/OpenCV/GateProject/forward2.avi                 5
#/home/xing/TesterCodes/OpenCV/GateProject/forward3.avi                 5
#/home/xing/TesterCodes/OpenCV/GateProject/forward12.avi                5
#/home/xing/TesterCodes/OpenCV/GateProject/move_towards.avi             11
#/home/xing/TesterCodes/OpenCV/GateProject/stand_still_far.avi          11
#/home/xing/TesterCodes/OpenCV/GateProject/close_up_gate.avi            11
#/home/xing/TesterCodes/OpenCV/GateProject/angle_move_towards.avi       11


    # image_prep.py tests
#            #col0 = img_prep.combineCol(sliced_blocks[:,0]) #show column
#            #cv2.imshow('leftCol',col0)
#            rowselect = 5
#            row0 = img_prep.combineRow(sliced_blocks[rowselect,:]) #show row
#            cv2.imshow('row',row0)
#            
#            #k_mean_whole = img_prep.kmeans(frame) # very slow
#            #cv2.imshow('original_k_mean',k_mean_whole)
#            k_mean_row = img_prep.kmeans(row0)
#            cv2.imshow('row_k_mean',k_mean_row)
#            k_mean_block = img_prep.kmeans(sliced_blocks[rowselect,0])
#            cv2.imshow('block_k_mean',k_mean_block)
#
#            blurrow = img_prep.blur(sliced_blocks[rowselect,:])
#            k_mean_blur = img_prep.kmeansList(blurrow)
#            k_mean_blurrow = img_prep.combineRow(k_mean_blur)
#            cv2.imshow('blurrow_k_mean',k_mean_blurrow)
#
#            #contour_k_mean_blurrow = img_prep.contour(k_mean_blurrow)
#            #cv2.imshow('contour_k_mean_blurrow',contour_k_mean_blurrow)
#            #contour_k_mean_row = img_prep.contour(k_mean_row)
#            #cv2.imshow('contour_k_mean_row',contour_k_mean_row)
#            contour_k_mean_blur = img_prep.contourList(k_mean_blur)
#            row_contour_k_mean_blur = img_prep.combineRow(contour_k_mean_blur)
#            cv2.imshow('row_contour_k_mean_blue',row_contour_k_mean_blur)