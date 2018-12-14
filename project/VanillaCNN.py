import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import keras
from matplotlib import pyplot as plt
import pandas as pd

####### HYPER PARAMETERS ##########

batchSz=100
shape=(128,128,4) #Images are originally 512 x 512 but we resize later on to make model faster
valSetRatio= 0.2 # Use 20% of training data as the validation set. 
Threshold= 0.05 ## probability threshold to predict a class as 'yes'
# Selected this after our first exploratory run and plotted threshold vs F1 score in this architecture
SEED =123 #For random initialization stuff
epochs=100
 
######### DATA IMPORT FUNCTIONS AND METHODS ############################
def getTrainDataset(path_to_trainData):
	data=pd.read_csv("".join(path_to_trainData+"/train.csv"))
	paths=[]
	labels=[]
	for name, label in zip(data['Id'], data['Target'].str.split(' ')):
		l=np.zeros(28)
		for group in label:
			l[int(group)]=1
		paths.append(os.path.join(path_to_trainData, name))
		labels.append(l)
	return np.array(paths), np.array(labels)

def getTestDataset(path_to_testData):
	data = pd.read_csv("".join(path_to_testData+ '/sample_submission.csv'))
	paths = []
	labels = []
	for name in data['Id']:
		l = np.ones(28)
		paths.append(os.path.join(path_to_testData, name))
		labels.append(l)

	return np.array(paths), np.array(labels)




class DataforModel(keras.utils.Sequence):
			
	def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False):
		self.paths, self.labels = paths, labels
		self.batch_size = batch_size
		self.shape = shape
		self.shuffle = shuffle
		self.use_cache = use_cache
		if use_cache == True:
			self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
			self.is_cached = np.zeros((paths.shape[0]))
		self.on_epoch_end()
	
	def __len__(self):
		return int(np.ceil(len(self.paths) / float(self.batch_size)))
	
	def __getitem__(self, idx):
		indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

		paths = self.paths[indexes]
		X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
		
		if self.use_cache == True:
			X = self.cache[indexes]
			for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
				image = self.__load_image(path)
				self.is_cached[indexes[i]] = 1
				self.cache[indexes[i]] = image
				X[i] = image
		else:
			for i, path in enumerate(paths):
				X[i] = self.__load_image(path)

		y = self.labels[indexes]
		
		return X, y
	
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.paths))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __iter__(self):
		for item in (self[i] for i in range(len(self))):
			yield item
			
	def __load_image(self, path):
		R = Image.open(path + '_red.png')
		G = Image.open(path + '_green.png')
		B = Image.open(path + '_blue.png')
		Y = Image.open(path + '_yellow.png')

		im = np.stack((
			np.array(R), 
			np.array(G), 
			np.array(B),
			np.array(Y)), -1)
		
		im= cv2.resize(im, self.shape[0], self.shape[1])
		im = np.divide(im, 255)
		return im  
	
################################################################3


########### CREATE MODEL ##################
set_random_seed(SEED)

# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

def f1(y_true, y_pred):
	"""
	Function to calculate the F1 score for accuracy evaluation
	"""
	y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), Threshold), K.floatx())
	tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
	tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
	fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
	fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = 2*p*r / (p+r+K.epsilon())
	f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
	return K.mean(f1)

def create_model(input_shape):
	dropRate = 0.25
	
	## LAYER 1
	initInput = Input(input_shape)
	x = Conv2D(16, (7, 7))(initInput)
	x = ReLU()(x)

	## LAYER 2
	x = BatchNormalization(axis=-1)(x)
	x = Conv2D(16, (5, 5))(x)
	x = ReLU()(x)

	## LAYER 3
	x = BatchNormalization(axis=-1)(x)
	x = Conv2D(32, (3, 3))(x)
	x = ReLU()(x)

	## LAYER 4
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropRate)(x)

	## LAYER 5
	x = Conv2D(16, (3, 3), padding='same')(x)
	x= ReLU()(x)

	## LAYER 6
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropRate)(x)
	
	## LAYER 7
	x = Conv2D(32, (3, 3))(x)
	x = ReLU()(x)
	
	## LAYER 8
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropRate)(x)
	
	## LAYER 9
	x = Conv2D(64, (1, 1))(x)
	x = ReLU()(x)
	
	## LAYER 10
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropRate)(x)
	
	## LAYER 11
	x = Conv2D(128, (1, 1))(x)
	x = ReLU()(x)
	
	## LAYER 12
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropRate)(x)

	## LAYER 13
	x = Flatten()(x)
	x = BatchNormalization(axis=-1)(x)
	x = Dense(28)(x)
	x = Activation('sigmoid')(x)
	
	model = Model(initInput, x)
	
	return model

model = create_model(shape)
model.compile(
	loss='binary_crossentropy',
	optimizer=Adam(1e-03),
	metrics=['acc',f1])

model.summary() #<--- Will print the architecture
######################################################

paths, labels = getTrainDataset()
keys = np.arange(paths.shape[0], dtype=np.int)  
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1-valSetRatio) * paths.shape[0])

pathsTrain = paths[0:lastTrainIndex]
labelsTrain = labels[0:lastTrainIndex]
pathsVal = paths[lastTrainIndex:]
labelsVal = labels[lastTrainIndex:]

tg = ProteinDataGenerator(pathsTrain, labelsTrain, batchSz, shape, use_cache=True, shuffle = False)
vg = ProteinDataGenerator(pathsVal, labelsVal, batchSz, shape, use_cache=True, shuffle = False)

# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
hist = model.fit_generator(
	tg,
	steps_per_epoch=len(tg),
	validation_data=vg,
	validation_steps=8,
	epochs=epochs,
	use_multiprocessing=False,
	workers=1,
	verbose=1,
	callbacks=[checkpoint])

#### PLOTTING RESULTS ################33
fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('Loss')
ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
ax[1].set_title('F1 accuracy')
ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
ax[0].legend()
ax[1].legend()

############ TEST DATASET PREDICTIONS FOR KAGGLE SUBMISSION ###########
pathsTest, labelsTest = getTestDataset()

testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
submit = pd.read_csv("".join(pathsTest + '/sample_submission.csv'))
P = np.zeros((pathsTest.shape[0], 28))
for i in range(len(testg)):
	images, labels = testg[i]
	score = bestModel.predict(images)
	P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score

PP = np.array(P)
prediction = []

for row in range(submit.shape[0]):
	
	str_label = ''
	
	for col in range(PP.shape[1]):
		if(PP[row, col] < T[col]):
			str_label += ''
		else:
			str_label += str(col) + ' '
	prediction.append(str_label.strip())
	
submit['Predicted'] = np.array(prediction)
submit.to_csv('submission_predictions.csv', index=False)