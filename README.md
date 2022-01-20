this is the OpenCV code I started over the break attempting to better analyze videos. Focus on analyzing gate video in the NCSU pool setting.

### How to interpret the images  
You will find a grayscale image made with completely black squares, some gray squares with highlighted edges of 2 colors. The brighter colored edge are filtered with certain parameters. This image is used to tell how and where it has been processed.
You will also find an 10x10 (or much smaller) image that is made from the brightest pixel in each square of the image described above. This image is used can be interpreted as a 2d array of different states (or also priority of each image region) avaliable for filtering.  

### Which still needs work
1. There is no algorithm for locating the gate. I think it can be solved by finding the maximum, or matching pattern, or a variety of ways that is common in computer science. But I dont have the knowledge to do so.
2. The current pipeline works well for NCSU Pool, which is clear and lots of edges visible. A different pipeline should be used for when the water is murky and blurrly
3. Theoretically this method can be used for different object detection other than gate, as long as a set of features can be extracted in building the pipeline.

image_prep.py - prepare an image and supply methods to modify an image/images
* slice - split the image into squares
* kmean - apply kmeans segmentation (kind of slow but works well in simplifying image by relative color)
* contour - canny contour finding

image_process.py - the labelling of images, give image meaning
* GateDetect
    * createBlocks - generate a 2d list of block properties (used to generate simple_img which shows the max color of the properties)\
    *    maskImg - process the image by slice -> blur -> kmean -> filter -> contour

image_util.py - random utilities for testing
* plotting

gate_detect.py - should just be a txt file that contain stuff for copy-paste
