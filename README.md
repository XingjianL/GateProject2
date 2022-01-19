this is the OpenCV code I started over the break attempting to better analyze videos. Focus on analyzing gate video in the NCSU pool setting.

image_prep.py - prepare an image and supply methods to modify an image/images\
    slice - split the image into squares\
    kmean - apply kmeans segmentation (kind of slow but works well in simplifying image by relative color)\
    contour - canny contour finding

image_process.py - the labelling of images, give image meaning\
    GateDetect\
        createBlocks - generate a 2d list of block properties (used to generate simple_img which shows the max color of the properties)\
        maskImg - process the image by slice -> blur -> kmean -> filter -> contour

image_util.py - random utilities for testing\
    plotting

gate_detect.py - should just be a txt file that contain stuff for copy-paste