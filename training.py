import matplotlib.pyplot as ply
import pandas as pd

import os, json
from glob import glob

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import nasnet

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model,load_model,Sequential

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,Flatten, Input
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger

import tensorflow as tf

# Check GPU exists
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Keras and Tensorflow version
print("Keras version:", tf.keras.__version__)
print("Tensorflow:", tf.__version__)

# Get breeding classes
breeds = open("breeds.txt").read().splitlines()

# Data
img_width, img_height = 224, 224

path = "./data/"
train_data_dir = "./data/train/"
valid_data_dir = "./data/valid/"

filenames = dict()
for breed in breeds:
	filenames[breed] = glob('./data/train/' + breed + '/*.jpg')

# Prepare for training

i = 0
df = dict()
for breed in breeds:
	df[breed] = pd.DataFrame(filenames[breed], columns=["filename"])
	df[breed]['class'] = pd.Series([i for x in range(len(df[breed].index))], index=df[breed].index)
	i += 1

train_df = dict()
valid_df = dict()
train_set_percentage = .9
for breed in breeds:
	train_df[breed] = df[breed][:int(len(df) * train_set_percentage)]
	valid_df[breed] = df[breed][int(len(df) * train_set_percentage):]

train_list = []
valid_list = []
for breed in breeds:
	train_list.append(train_df[breed])
	valid_list.append(valid_df[breed])

df_new_train = pd.concat(train_list)
df_new_valid = pd.concat(valid_list)

df_train = df_new_train.sample(frac=1).reset_index(drop=True)
df_valid = df_new_valid.sample(frac=1).reset_index(drop=True)

train_filenames_list = df_train["filename"].tolist()
train_labels_list = df_train["class"].astype('int32').tolist()

valid_filenames_list = df_valid["filename"].tolist()
valid_labels_list = df_valid["class"].astype('int32').tolist()

num_classes = len(breeds)

# Pipeline
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [img_height, img_width])
  image_resized = tf.ensure_shape(image_resized ,shape=(img_height, img_width, 3))
  label = tf.one_hot(label, num_classes)
  return image_resized, label

filenames = tf.constant(train_filenames_list)
labels = tf.constant(train_labels_list)

val_filenames = tf.constant(valid_filenames_list)
val_labels = tf.constant(valid_labels_list)

# Training parameters
batch_size = 32
epochs = 1000
learning_rate = 0.001
train_steps = int(9200 / batch_size) #total trains set / batch_size
val_steps = int(1000 / batch_size)

## Assumbling pipeline

train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

valid_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
valid_dataset = valid_dataset.map(_parse_function)
valid_dataset = valid_dataset.repeat()
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)

# Model
base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()

# add a global spatial average pooling layer
x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

x = Dropout(0.3)(x)

predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional VGG16 layers
for layer in base_model.layers:
    print(layer.name)
    layer.trainable = False

# opt = tf.train.AdamOptimizer(learning_rate = 0.001)

opt = tf.keras.optimizers.Adam(lr=learning_rate)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('./checkpoints/new_weights_{epoch:02d}_{val_acc:.2f}.hdf5', verbose=1, save_best_only=True, mode='auto')

# # # Train the model with validation 
history = model.fit( train_dataset, steps_per_epoch = train_steps,
                   epochs = epochs,
                   verbose=1,
                   validation_data = valid_dataset,
                   validation_steps = val_steps,
                   callbacks=[checkpoint])

