import csv
from collections import defaultdict
import numpy as np
from PIL import Image

"""
NOTE: It takes a long time to read every image file. If you have any recommendations on that, pls let me know!!
-Pinar Demetci
"""

def next_batch(batchSize, images, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(labels))
    np.random.shuffle(idx)
    idx = idx[:batchSize]
    images_batch = [images[i] for i in idx]
    labels_batch = [labels[i] for i in idx]

    return np.asarray(images_batch), np.asarray(labels_batch)


def load_images(path_to_images,image_ids):
	"""
	Returns an array of arrays for each training example such that:
	[ [red channel, 1 by 262144],
	  [blue channel, 1 by 262144],
	  [green channel, 1 by 262144],
	  [yellow channel, 1 by 262144]	]
	"""
	all_images=[] # should this be an np array? Currently, it is a list that holds np arrays of each training example
	for ID in image_ids:
		red_image= image_to_array(''.join(path_to_images+ID+'_red.png'))
		blue_image= image_to_array(''.join(path_to_images+ID+'_blue.png'))
		green_image= image_to_array(''.join(path_to_images+ID+'_green.png'))
		yellow_image= image_to_array(''.join(path_to_images+ID+'_yellow.png'))
		all_images.append(np.stack((red_image, blue_image, green_image, yellow_image)))
	return all_images

def load_training_labels(IDs, labels):
	"""
	Returns a list of labels for all training examples. Labels are in one hot format 
	"""
	y=[]
	for l in labels:
		l_list=l.split(' ')
		l_list = [ int(x) for x in l_list ] #they are in string format but we want them in int format for turning into one hot vectors
		y.append(to_onehot(l_list))
	return y

def image_to_array(image_file):
	"""
	Helper function:
	Reads in a single image and flattens it, returning a 1 by 262144.
	262144 is due to 512 x 512 (image size)
	"""
	with open(image_file,'rb') as f:
		pil_img=Image.open(f).convert(mode='L')
		pil_img=np.array(pil_img).reshape(1,262144)
		return pil_img

def to_onehot(numbers):
	"""
	Helper function:
	Takes in the labels for each training example and turns into a one-hot vector for 28 labels
	Note: I could not figure out how to use the tf.one_hot() function for multiple labels per example
	Feel free to change this function
	"""
	ls=np.zeros([28], dtype=np.int32)
	for number in numbers:
		ls[number]=1
	return ls

def load_IDsLabels(path_to_labels, label_filename):
	"""
	Returns the IDs for training examples and labels for the corresponding ID.
	For training dataset, label_file will be 'train.csv'
	For test dataset, label_file will be 'sample_submission.csv'
	These csv files have two columns: First column correspond to image filenames or "ID"s for each example
	Second column corresponds to labels for proteins e.g. 0 5 28 
	"""
	label_file="".join(path_to_labels+label_filename)
	reader=csv.reader(open(label_file,'r'))
	next(reader) # Skip the header row
	ID_label={}
	for row in reader:
		ID, labels=row 
		ID_label[ID]=labels # Dictionary, for each 'ID' (which also gives the corresponding image filename), the labels for that ID
	return ID_label.keys(), ID_label.values() #So that we can use 'keys' as IDs for loading images and 'values' to get lables turn into one hot vectors


if __name__ == "__main__":
	path_to_labels="labels/" # Modify for your own use! Path to the 'train.csv' file
	path_to_trainImages="train/" #Modify for your own use! Path to the training images
	path_to_testImages="test/" #Modify for your own use! Path to the test images

	test_IDs, _ = load_IDsLabels(path_to_labels,"sample_submission.csv")
	train_IDs, train_labels= load_IDsLabels(path_to_labels,"train.csv")

	X_train=load_images(path_to_trainImages, train_IDs)
	y_train=load_training_labels(train_IDs, train_labels)

	X_test=load_images(path_to_testImages, test_IDs)

	batched_images, batched_labels=next_batch(100, X_train, y_train)