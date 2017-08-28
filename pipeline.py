import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip
import time


class ParamLoader(object):
	""" Loads trained params """
	def __init__(self, pickle_file):
		super(ParamLoader, self).__init__()	
		d = pickle.load(open(pickle_file, 'rb'))
		self.clf = d['clf']
		self.scaler = d['scaler']
		self.orientations = d['orientations']
		self.pixels_per_cell = d['pixels_per_cell']
		self.cells_per_block = d['cells_per_block']
		self.hog_channel = d['hog_channel']
		self.spatial_size = d['spatial_size']
		self.hist_bins = d['hist_bins']
		self.cspace = d['cspace']
		self.extract_bin_spatial=d['extract_bin_spatial']
		self.extract_hog=d['extract_hog']
		self.extract_color_hist=d['extract_color_hist']
		self.num_of_pictures = 0
		self.prev_labels = None
		self.heatmap = None


class ColorSpaceError(Exception):
	pass


param_loader = ParamLoader('train_9_16_2_0_16_32_YUV.p')


def color_hist(img, nbins=32, range=(0,255)):
	""" get histogram features, the input image 
		should be 3-channel image 
	"""
	color_hist1 = np.histogram(img[:,:,0], nbins, range=range)
	color_hist2 = np.histogram(img[:,:,1], nbins, range=range)
	color_hist3 = np.histogram(img[:,:,2], nbins, range=range)
	features = np.concatenate((color_hist1[0], color_hist2[0], color_hist3[0]))
	return features


def bin_spatial(img, size=(32,32)):
	""" perform spatial binning on the image and 
		convert to feature vector """
	return np.resize(img, size).ravel()


def get_hog_features(img, orientations=9, pixels_per_cell=8, cells_per_block=2, 
                     visualize=False, feature_vector=True):
	"""	get hog features from the input image
		the input image should be single channel 
	"""
	if visualize:
		features, hog_img = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell,pixels_per_cell), 
			cells_per_block=(cells_per_block,cells_per_block), visualise=True, feature_vector=feature_vector)
		return features, hog_img
	else:
		features = hog(img, orientations=orientations, pixels_per_cell=(pixels_per_cell,pixels_per_cell), 
			cells_per_block=(cells_per_block,cells_per_block), feature_vector=feature_vector)
		return features


def extract_features(imgs, cspace='RGB', extract_bin_spatial=True, extract_hog=True, extract_color_hist=True, 
		orientations=9, pixels_per_cell=8, cells_per_block=2, hog_channel=-1, spatial_size=(32,32), hist_bins=32):
	""" extract features for the image list """
	features = []
	for img in imgs:
		file_features = []

		if cspace == 'RGB':
			feature_image = np.copy(img)
		else:
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
			else:
				raise ColorSpaceError('Invalid colorspace')

		# extract color histogram features
		if extract_color_hist:
			color_hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(color_hist_features)

		# extract spatial binning features
		if extract_bin_spatial:
			bin_spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(bin_spatial_features)

		# extract HOG features
		if extract_hog:
			# if -1 is specified for hog_channel
			# get hog features for all channel
			# otherwise, the hog features for the specified channel
			if hog_channel == -1:
				hog_features = []
				for c in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,c], orientations, pixels_per_cell, cells_per_block))
				file_features.append(np.ravel(hog_features))
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orientations, pixels_per_cell, cells_per_block)
				file_features.append(hog_features)
		features.append(np.concatenate(file_features))
	return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	"""	calculate the windows which scans the image	"""

	# if start/stop coordinates are not specified, use image size
	x_start_stop[0] = 0 if x_start_stop[0] is None else x_start_stop[0]
	x_start_stop[1] = img.shape[1] if x_start_stop[1] is None else x_start_stop[1]
	y_start_stop[0] = 0 if y_start_stop[0] is None else y_start_stop[0]
	y_start_stop[1] = img.shape[0] if y_start_stop[1] is None else y_start_stop[1]


	# calculate the spans
	x_span = x_start_stop[1] - x_start_stop[0]
	y_span = y_start_stop[1] - y_start_stop[0]

	# calculate step sizes
	x_step_size = np.multiply(xy_window[0], 1 - xy_overlap[0]).astype(int)
	y_step_size = np.multiply(xy_window[1], 1 - xy_overlap[1]).astype(int)

	# calculate number of windows
	x_num_windows = ((x_span - xy_window[0])/x_step_size + 1).astype(int)
	y_num_windows = ((y_span - xy_window[1])/y_step_size + 1).astype(int)

	windows = []
	for i in range(x_num_windows):
		x_start = x_start_stop[0] + i*x_step_size
		for j in range(y_num_windows):
			y_start = y_start_stop[0] + j*y_step_size
			windows.append(((x_start, y_start), (x_start+xy_window[0], y_start+xy_window[1])))
	return windows


def search_windows(img, windows, clf, scaler, cspace='RGB', extract_bin_spatial=True, extract_hog=True, extract_color_hist=True, 
		orientations=9, pixels_per_cell=8, cells_per_block=2, hog_channel=-1, spatial_size=(32,32), hist_bins=32):
	""" search cars from the passed in windows """

	car_windows = []
	for window in windows:
		sub_image = cv2.resize(img[window[0][1]: window[1][1], window[0][0]: window[1][0]], (64,64))
		features = extract_features([sub_image], cspace, extract_bin_spatial, extract_hog, extract_color_hist,
			orientations, pixels_per_cell, cells_per_block, hog_channel, spatial_size, hist_bins)
		t=np.array(features[0]).reshape(1, -1)
		scaled_features = scaler.transform(t)
		pred = clf.predict(scaled_features)
		if pred == 1:
			car_windows.append(window)
	return car_windows


def draw_boxes(img, boxes, color=(0,0,255), thickness=6):
	""" draw boxes on the input image with the specified color 
		and thickness. The boxes format is [((x1,y1), (x2,y2)), ((x3,y3), (x4,y4))] 
	"""
	img_copy = np.copy(img)
	for box in boxes:
		print(box[0])
		print(box[1])
		cv2.rectangle(img_copy, box[0], box[1], color, thickness)
	return img_copy


def add_heat(heatmap, boxes):
	for box in boxes:
		# box => ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	return heatmap


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap


def draw_box_with_labels(img, lables):
	cars = lables[1]
	img_copy = np.copy(img)
	for car in range(1, cars+1):
		nonzero = (lables[0] == car).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		cv2.rectangle(img_copy, box[0], box[1], (0,0,255), 6)
	return img_copy


def normalize(car_features, not_car_features):
	X = np.vstack((car_features, not_car_features)).astype(np.float64)
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))
	return X_scaler, scaled_X, y


def train_big_dataset(orientations=9, pixels_per_cell=8, cells_per_block=2, spatial_size = (16, 16),
		hist_bins = 32, hog_channel=-1, cspace='YUV', extract_bin_spatial=True, extract_hog=True, extract_color_hist=True,
		pickle_file="train_1.p"):

	car_image_paths = glob.glob('vehicles/*/*.png')
	not_car_image_paths = glob.glob('non-vehicles/*/*.png')

	car_images = [mpimg.imread(car_image_path) for car_image_path in car_image_paths]
	not_car_images = [mpimg.imread(not_car_image_path) for not_car_image_path in not_car_image_paths]

	fs, hog_img = get_hog_features(car_images[0][:,:,0], orientations=orientations, pixels_per_cell=pixels_per_cell,
		cells_per_block=cells_per_block, visualize=True)
	# plt.figure(1)
	# plt.subplot(2,2,1)
	# plt.imshow(car_images[0])
	# plt.subplot(2,2,2)
	# plt.imshow(hog_img, cmap='gray')
	# plt.show()

	car_features = extract_features(car_images, orientations=orientations, pixels_per_cell=pixels_per_cell,
		cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
		extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	not_car_features = extract_features(not_car_images, orientations=orientations, pixels_per_cell=pixels_per_cell,
		cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
		extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	scaler, X, y = normalize(car_features, not_car_features)
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
	clf = train(X_train, y_train)
	accu = test(clf, X_test, y_test)

	if pickle_file is None:
		pickle_file = 'train_orientations_{}_pixels_per_cell_{}_hog_channel_{}_accu_{}.p'.format(
			orientations, pixels_per_cell, hog_channel, accu)

	d = dict()
	d['clf'] = clf
	d['scaler'] = scaler
	d['orientations'] = orientations
	d['pixels_per_cell'] = pixels_per_cell
	d['cells_per_block'] = cells_per_block
	d['spatial_size'] = spatial_size
	d['hist_bins'] = hist_bins
	d['hog_channel'] = hog_channel
	d['cspace'] = cspace
	d['extract_bin_spatial'] = extract_bin_spatial
	d['extract_hog'] = extract_hog
	d['extract_color_hist'] = extract_color_hist

	pickle.dump(d, open(pickle_file, "wb"))


def train(X_train, y_train):
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	return clf

def train_all():
	for orientations in range(8, 13, 2):
		for pixels_per_cell in range(8, 17, 4):
			for hog_channel in range(2, 3):
				for cspace in ['HSV','LUV','HLS','YUV','YCrCb']:
					train_big_dataset(orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=2, 
						spatial_size = (16, 16), hist_bins = 32, hog_channel=hog_channel, cspace=cspace, extract_bin_spatial=False, 
						extract_hog=True, extract_color_hist=False, pickle_file=None)


def test(clf, X_test, y_test):
	accu = clf.score(X_test, y_test)
	print("Test accuracy: {}".format(accu))
	return accu


def process_image(img):
	# now = time.time()
	orig_image = img
	# mpimg.imsave('/Users/luyaoli/video_images/{}.jpg'.format(now), orig_image)
	# return orig_image
	test_image = orig_image.astype(np.float32)/255

	clf = param_loader.clf
	scaler = param_loader.scaler
	orientations = param_loader.orientations
	pixels_per_cell = param_loader.pixels_per_cell
	cells_per_block = param_loader.cells_per_block
	hog_channel = param_loader.hog_channel
	spatial_size = param_loader.spatial_size
	hist_bins = param_loader.hist_bins
	cspace = param_loader.cspace
	extract_bin_spatial=param_loader.extract_bin_spatial
	extract_hog=param_loader.extract_hog
	extract_color_hist=param_loader.extract_color_hist
	
	xy_window = (64, 64)
	xy_overlap = (0.9, 0.9)
	x_start_stop = [img.shape[1]//2, None]
	y_start_stop = [400, 464]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows = search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	xy_window = (96, 96)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [img.shape[1]//2, None]
	y_start_stop = [400, 592]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	
	xy_window = (128, 128)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [img.shape[1]//2, None]
	y_start_stop = [400, 656]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	xy_window = (256, 256)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [img.shape[1]//2, None]
	y_start_stop = [400, 656]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)
	param_loader.num_of_pictures += 1
	
	if param_loader.heatmap is None:
		param_loader.heatmap = np.zeros_like(test_image[:,:,0]).astype(np.float)
	add_heat(param_loader.heatmap, hot_windows)

	if param_loader.prev_labels is None or param_loader.num_of_pictures == 8:
		if param_loader.prev_labels is None:
			updated_heat = apply_threshold(param_loader.heatmap, 2)
		else:
			updated_heat = apply_threshold(param_loader.heatmap, 10)
		labels = label(updated_heat)
		ret = draw_box_with_labels(orig_image, labels)
		param_loader.prev_labels = labels
		param_loader.num_of_pictures = 0
		param_loader.heatmap = None
		return ret
	ret = draw_box_with_labels(orig_image, param_loader.prev_labels)
	return ret


def run():
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
	x_start_stop = [None, None]
	y_start_stop = [350, 650]
	pickle_file = "train_{}_{}_{}_{}_{}_{}_{}.p".format(
		orientations, pixels_per_cell, cells_per_block, hog_channel, spatial_size[0], hist_bins, cspace)

	#car_image_paths = glob.glob('vehicles_smallset/*/*.jpeg')
	#not_car_image_paths = glob.glob('non-vehicles_smallset/*/*.jpeg')
	
	# train_big_dataset(orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, 
	# 	spatial_size=spatial_size, hist_bins=hist_bins, hog_channel=hog_channel, cspace=cspace, 
	# 	extract_bin_spatial=extract_bin_spatial, extract_hog=extract_hog, extract_color_hist=extract_color_hist,
	# 	pickle_file=pickle_file)
	
	#orig_image = mpimg.imread('/Users/luyaoli/video_images/1503846781.78735.jpg')
	orig_image = mpimg.imread('/Users/luyaoli/video_images/1503847155.767736.jpg')
	#orig_image = mpimg.imread('/Users/luyaoli/video_images/1503847421.799004.jpg')
	#orig_image = mpimg.imread('/Users/luyaoli/video_images/1503847129.805015.jpg')
	test_image = orig_image.astype(np.float32)/255

	d = pickle.load(open(pickle_file, 'rb'))
	clf = d['clf']
	scaler = d['scaler']

	windows = []
	# xy_window = (32, 32)
	# xy_overlap = (0.5, 0.5)
	# x_start_stop = [None, None]
	# y_start_stop = [350, 500]
	# windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	# hot_windows = search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	#  	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	#  	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)
	
	xy_window = (64, 64)
	xy_overlap = (0.9, 0.9)
	x_start_stop = [None, None]
	y_start_stop = [400, 464]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows = search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	xy_window = (96, 96)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [None, None]
	y_start_stop = [400, 592]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	
	xy_window = (128, 128)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [None, None]
	y_start_stop = [400, 656]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	xy_window = (256, 256)
	xy_overlap = (0.75, 0.75)
	x_start_stop = [None, None]
	y_start_stop = [400, 656]
	windows = slide_window(test_image, x_start_stop, y_start_stop, xy_window, xy_overlap)
	hot_windows += search_windows(test_image, windows, clf, scaler, orientations=orientations, pixels_per_cell=pixels_per_cell,
	 	cells_per_block=cells_per_block, cspace=cspace, extract_bin_spatial=extract_bin_spatial,
	 	extract_hog=extract_hog, extract_color_hist=extract_color_hist, hist_bins=hist_bins, spatial_size=spatial_size)

	
	heat = np.zeros_like(test_image[:,:,0]).astype(np.float)
	updated_heat = add_heat(heat, hot_windows)
	updated_heat = apply_threshold(heat, 2)
	labels = label(updated_heat)
	ret = draw_box_with_labels(orig_image, labels)

	# plt.subplot(2,2,3)
	plt.imshow(orig_image)
	plt.savefig('examples/example.jpg')

	# xy_window = (64, 64)
	# xy_overlap = (0.5, 0.5)
	# image_draw = find_cars(test_image, scaler, clf, x_start_stop, y_start_stop, xy_window, xy_overlap)
	# plt.subplot(2,2,3)
	# plt.imshow(image_draw)
	# plt.show()


def video():
	white_output = './project_video_output2.mp4'
	clip1 = VideoFileClip("./project_video.mp4")
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)


def show_hog_images():
	pickle_file = "train_4.p"

	car_image_paths = glob.glob('vehicles/*/*.png')
	not_car_image_paths = glob.glob('non-vehicles/*/*.png')
	
	cnt = 1
	for c in range(0,4):
		car_path = car_image_paths[c]
		img = mpimg.imread(car_path)
		_, hog_img = get_hog_features(img[:,:,0], orientations=9, pixels_per_cell=8, cells_per_block=2, 
                 visualize=True, feature_vector=True)
		plt.subplot(4,4, cnt)
		plt.imshow(img)
		cnt += 1
		plt.subplot(4,4, cnt)
		plt.imshow(hog_img, cmap='gray')
		cnt += 1
	for c in range(0,4):
		notcar_path = not_car_image_paths[c]
		img = mpimg.imread(notcar_path)
		_, hog_img = get_hog_features(img[:,:,0], orientations=9, pixels_per_cell=8, cells_per_block=2, 
                 visualize=True, feature_vector=True)
		plt.subplot(4,4, cnt)
		plt.imshow(img)
		cnt += 1
		plt.subplot(4,4, cnt)
		plt.imshow(hog_img, cmap='gray')
		cnt += 1
	plt.savefig('examples/HOG_examples_channel_0.png')
	


if __name__ == '__main__':
	#train_all()
	#video()
	run()
	#show_hog_images()


