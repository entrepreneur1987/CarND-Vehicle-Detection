
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/notcar.png
[image3]: ./examples/HOG_examples_channel_0.jpg
[image4]: ./examples/hot_windows.jpg
[image5]: ./examples/windows.jpg
[image6]: ./examples/test_classifier.jpg
[image7]: ./examples/heatmap.jpg
[image8]: ./examples/example.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car][image1] ![Not car][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, including include color histogram/spatial binning or HOG alone, and different params for HOG. I found out that color histogram/spatial binning doesn't help much in the final result. I tried many different combinations and here are my final choices:

```
orientations = 11  # HOG orientations
pixels_per_cell = 16 # HOG pixels per cell
cells_per_block = 2 # HOG cells per block
hog_channel = -1 # Can be 0, 1, 2, or -1
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32
cspace = 'YUV'
extract_bin_spatial=False
extract_hog=True
extract_color_hist=False
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn SVM package, with original car/notcar datasets. The test accuracy I got is 97.8%. The code is between line 242 and 254.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used three different window size listed below:

| Window Size   | 
|---------------|
| (64, 64)      | 
| (96, 96)      | 
| (128, 128)    | 
| (256, 256)    | 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:
![Windows][image5]![Found windows][image4]![Output image from pipeline][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example frame and its corresponding heatmaps:

![Original image][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![Heatmap][image7]

### Here are the resulting bounding boxes are drawn onto this frame:
![Final result][image4]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Sometimes some false positive cannot be filtered out. Or sometimes two boxes on one car. This is caused by threshold set on the heatmap. Different images may have different heapmap threshold requirements, which is hard to adjust as setting a threshold is very exprimental.

The approach I took is accumulating multiple frames (10 I set), and filter by a larger value. It works pretty well, except a few occations that a small car is not detected. The reason behind it is a false positive is unlikely to happen in multiple frames, as a result, we can reduce the false negative values. 

