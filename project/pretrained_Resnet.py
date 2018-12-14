import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

path_to_train = '../input/train/'
data = pd.read_csv('../input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


def create_train(dataset_info, batch_size, shape):
    while True:
        random_indexes = np.random.choice(len(dataset_info), batch_size)
        batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
        batch_labels = np.zeros((batch_size, 28))
        for i, idx in enumerate(random_indexes):
            image = load_image(dataset_info[idx]['path'], [shape[0], shape[1]])
            batch_images[i] = image
            batch_labels[i][dataset_info[idx]['labels']] = 1
        yield batch_images, batch_labels


def load_image(path,size):
    image_red_ch = np.asarray(Image.open(path + '_red.png').resize(size))
    image_green_ch = np.asarray(Image.open(path + '_green.png').resize(size))
    image_blue_ch = np.asarray(Image.open(path + '_blue.png').resize(size))
    image_yellow_ch = np.asarray(Image.open(path + '_yellow.png').resize(size))
    image = np.stack(
        (image_red_ch/255,
        image_green_ch/255,
        image_blue_ch/255,
        image_yellow_ch/255), -1
    )
    return image


# create train datagen
train_datagen = create_train(
    train_dataset_info, 5, (256,256,4))

images, labels = next(train_datagen)

fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))


from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import keras


# Create a model
n_out = 28
input_tensor = Input(shape=(256, 256, 4))
x = BatchNormalization()(input_tensor)
x = Dropout(0.5)(x)
# convert channel from 4 to 3
x = Conv2D(3, kernel_size=(1,1), strides=(1,1), activation=None)(x)
x = BatchNormalization()(x)

# get some layers from pre-trained model
base_model = ResNet50(include_top=False, weights='imagenet')
# base_output = base_model.get_layer(index=18).output
# base_input = base_model.input
# base_model = Model(inputs=base_input, outputs=base_output)
x = base_model(x)
x = Flatten()(x)

x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
output = Dense(n_out, activation='sigmoid')(x)
model = Model(input_tensor, output)
model.summary()

# loss and metrics
THRESHOLD = 0.5


def precision(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    return tp / (tp + fp + K.epsilon())


def recall(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    return tp / (tp + fn + K.epsilon())


def f1(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def log(x):
    # helper function. Stable log
    return tf.log(tf.maximum(x, 1e-5))


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(-log(f1))


epochs = 20; batch_size = 32

# split and suffle data
np.random.seed(2018)
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes = indexes[:27500]
valid_indexes = indexes[27500:]

# create train and valid datagens
train_generator = create_train(train_dataset_info[train_indexes], batch_size, (256,256,4))
validation_generator = create_train(train_dataset_info[valid_indexes], batch_size, (256,256,4))

# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(len(train_indexes)/batch_size),
    validation_data=next(validation_generator),
    validation_steps=int(len(valid_indexes)/batch_size),
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list)

fig, ax = plt.subplots(2, 2, figsize=(15,10))
ax[0][0].set_title('loss')
ax[0][0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0][0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[0][1].set_title('f1')
ax[0][1].plot(history.epoch, history.history["f1"], label="Train f1")
ax[0][1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
ax[1][0].set_title('precision')
ax[1][0].plot(history.epoch, history.history["precision"], label="Train precision")
ax[1][0].plot(history.epoch, history.history["val_precision"], label="Validation precision")
ax[1][1].set_title('recall')
ax[1][1].plot(history.epoch, history.history["recall"], label="Train recall")
ax[1][1].plot(history.epoch, history.history["val_recall"], label="Validation recall")

ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()

submit = pd.read_csv('../input/sample_submission.csv')

%%time
predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = load_image(path, (256,256))
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=THRESHOLD]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)


submit['Predicted'] = predicted
submit.to_csv('submission_ResNet50-finetune.csv', index=False)