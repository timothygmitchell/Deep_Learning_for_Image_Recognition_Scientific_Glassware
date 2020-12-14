#!/usr/bin/env python
# coding: utf-8

# # Deep Learning for Image Classification
# 
# ### Timothy Mitchell

# #### Helpful Keyboard Shortcuts in Jupyter Notebooks

# In[1]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(filename="/home/tim/Pictures/JupyterNotebookShortcuts.png", width=500, height=450)


# # Sorting Data
# 
# #### Step 1
# 
# Unzip image archives with the Bash command `7z e [Filename].zip`.
# 
# Download images into `~/Documents/Py/Flasks/images` and sort into directories.

# In[2]:


import os
os.chdir("/home/tim/Documents/Py/Flasks/") # change directory

get_ipython().system('tree -d images/')


# #### Step 2
# Compress images and strip filenames of spaces using the following .R script.

# 
# ```
# setwd("~/Documents/Py/Flasks/images/A/")        # set working directory
# 
# # install.packages("BiocManager")
# # BiocManager::install("EBImage")
# library(EBImage)
# 
# images <- list.files()                          # extract the names of the images
# n <- length(images)
# 
# for (i in 1:n) {                                # script to loop through directory, compressing images
# 
#   cat("Compressing image", i, "of", n, "\n")    # print progress at each iteration
# 
#   x <- readImage(images[i], type = "JPEG")      # read in image
# 
#   writeImage(x, gsub(" ", "", images[i]),       # compress image and trim white space from filename
#              type = "JPEG", quality = 80L)      # 80% compression
# 
#   if (images[i] != gsub(" ", "", images[i]))    # if old image still exists,
#     { unlink(images[i]) }                       # then delete old image from directory
# 
# }                                               # repeat for B, C, D, E
# ```

#  Alternatively:
#  - Python: `PIL`.
#  - Bash: `ImageMagick`.
#  - Windows: `Mass Image Compressor 3.1`.

# #### Step 3
# Randomize filenames in each directory.
# 
# Unfortunately, the `validation_split` parameter in `Keras` does NOT shuffle the data, as one might expect. From the documentation:
# 
# >validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.
# 
# As a workaround, we can assign each image a random string as a filename.

# In[3]:


from string import ascii_lowercase
from random import choice, randint, random

def randomize_filenames(dir):
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if os.path.isfile(path):
            newpath = os.path.join(dir, "".join([choice(ascii_lowercase) for _ in range(randint(5,8))])+".jpg")
            os.rename(path, newpath)


# In[4]:


for d in ["/home/tim/Documents/Py/Flasks/images/"+s for s in os.listdir("/home/tim/Documents/Py/Flasks/images/")]:
    randomize_filenames(d)


# # Exploratory Data Analysis

# In[5]:


# extract filenames

import glob
A = glob.glob("images/A/*.jpg")
B = glob.glob("images/B/*.jpg")
C = glob.glob("images/C/*.jpg")
D = glob.glob("images/D/*.jpg")
E = glob.glob("images/E/*.jpg")
F = glob.glob("images/F/*.jpg")
G = glob.glob("images/G/*.jpg")
#H = glob.glob("images/H/*.jpg")
#I = glob.glob("images/I/*.jpg")
J = glob.glob("images/J/*.jpg")
#Apple = glob.glob("images/Apple/*.jpg")
#Oil = glob.glob("images/Oil/*.jpg")
Other = glob.glob("images/Other/*.jpg")


# In[6]:


from matplotlib import pyplot
from matplotlib.image import imread
import numpy as np

# Function to preview 10 random images
def showimages(sourcefolder):
    fig = pyplot.figure(figsize=(18, 4.3))
    columns = 5
    rows = 2
    for i in range(1, columns*rows + 1):
        filename = sourcefolder[i]
        image = imread(filename)
        fig.add_subplot(rows, columns, i)
        pyplot.imshow(image)
    pyplot.show()
    return


# ## Flask A

# In[7]:


showimages(A)


# ## Flask B

# In[8]:


showimages(B)


# ## Flask C

# In[9]:


showimages(C)


# ## Flask D

# In[10]:


showimages(D)


# ## Flask E

# In[11]:


showimages(E)


# ## Flask F

# In[12]:


showimages(F)


# ## Flask G

# In[13]:


showimages(G)


# ## Flask J

# In[14]:


showimages(J)


# ## Other
# 
# Images from the Vector-LabPics data set (University of Toronto).
# 
# Glassware including flasks, bowls, beakers, dishes, and pipettes from chemistry labs and other contexts. 
# 
# A variety of materials are represented (liquid, solid, foam, suspension, powder, gel, granular, vapor).
# 
# Images vary in resolution, aspect ratio, and presence of artifacts (such as text).

# In[15]:


fig=pyplot.figure(figsize=(18, 14))
columns = 6
rows = 6
for i in range(1, columns*rows +1):
    filename = Other[i]
    image = imread(filename)
    fig.add_subplot(rows, columns, i)
    pyplot.imshow(image)
pyplot.show()


# # Image Preprocessing

# ## Downsizing Images

# Images are downsized to a standard aspect ratio (150 × 150).

# In[16]:


from keras.preprocessing.image import load_img

fig=pyplot.figure(figsize=(15, 4))

image = load_img(A[30])                            # original image
fig.add_subplot(1, 2, 1)
pyplot.imshow(image)

image = load_img(A[30], target_size=(150, 150))    # downsized image
fig.add_subplot(1, 2, 2)
pyplot.imshow(image)

pyplot.show()


# ## Converting Images to Arrays

# In[17]:


from keras.preprocessing.image import array_to_img, img_to_array

image = load_img(A[30], target_size=(150, 150))

x = img_to_array(image)  # encode the image as an array

x


# In[18]:


x.shape


# `x` has three dimensions with shape `(150, 150, 3)`. 
# 
#  1. The first dimension gives the horizontal position of a pixel
#  2. The second dimension gives the vertical position of a pixel
#  3. The third dimension gives the intensities of the red, green, and blue color channels, respectively

# ## Min-Max Normalization

# In[19]:


x = x/255

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

x # pixel values are bounded from 0 to 1


# ## Reshaping Arrays for `Keras`

# `Keras` also requires us to specify the batch size (how many images are passed to the model at a time).
# 
# As a result, `Keras` inputs are 4-dimensional arrays with shape (batch size, image width, image height, color).
# 
# Note: the array shape depends on the backend (in our case, `Tensorflow`).

# In[20]:


x = x.reshape((1,) + x.shape)  # this is a NumPy array with shape (1, 150, 150, 3)

x.shape # new dimensions


# During model training, the `ImageDataGenerator()` function and `.flow_from_directory()` method downsize images and convert them to NumPy arrays as described here.

# # Data Augmentation

# In[21]:


from keras.preprocessing.image import ImageDataGenerator

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.15)

# this is the augmentation configuration we will use for testing:
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.15)


# ### Example

# In[22]:


os.mkdir("temp/") # temporary directory

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `temp/` directory

i = 0
for batch in train_datagen.flow(x, batch_size=1,
                                save_to_dir='temp', 
                                save_prefix='flask', 
                                save_format='jpeg'):
    i += 1
    if i > 21:
        break


# ### Augmented Data

# In[23]:


temp = glob.glob("temp/*.jpeg")
temp
fig=pyplot.figure(figsize=(13, 6))
columns = 6
rows = 3
for i in range(1, columns*rows +1):
    filename = temp[i]
    image = imread(filename)
    fig.add_subplot(rows, columns, i)
    pyplot.imshow(image)
pyplot.show()


# In[24]:


import shutil
shutil.rmtree("temp/") # remove directory


# # Batch Size and `flow_from_directory()`

# In[25]:


batch_size = 8

train_generator = train_datagen.flow_from_directory(
        'images',  # this is the target directory
        seed = 1234,
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='training')

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        'images',
        seed = 1234,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation')


# # Check for Data Leakage

# In[26]:


filecount = sum([len(files) for r, d, files in os.walk("images/")]) # number of images
filecount


# In[27]:


len(train_generator.filepaths) + len(validation_generator.filepaths) == filecount


# The number of files in the directory tree is exactly equal to the sum of the number of images in the training and validation data.

# In[28]:


def intersection(list1, list2): 
    return list(set(list1) & set(list2))

intersection(train_generator.filepaths, validation_generator.filepaths)


# The intersection of `train_generator.filepaths` and `validation_generator.filepaths` is the null set, so we can say with confidence that the model was validated on unseen data.

# # Swish Activation Function

# ### Swish

# In[29]:


from keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})


# ### Hard Swish

# In[30]:


import tensorflow as tf

def hswish(x):
    return (x * tf.keras.activations.relu(x+3, max_value=6) * 0.16666667)

get_custom_objects().update({'hswish': Activation(hswish)})


# In the end, Swish and Hard Swish were found to be too slow, and they were abandoned.

# # Convolutional Neural Network

# In[31]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense


# `Sequential` is the basic framework for fitting neural networks in `Keras`.
# 
# `Conv2D` = convolution layers. These layers apply filters to the input.
# 
# `MaxPooling2D` = pooling layers. These layers combat overfitting and boost speed by downscaling inputs without losing too much information. The most common choice of pool size is 2×2.
# 
# `Activation` = activation layer. These layers passes important features to the next layer. (When the activation layer is preceded by a convolution layer, the activation function captures a feature map; when the activation function comes at the end of the network, the activation function captures class probabilities.) The most common activation function used to be sigmoid, until it was discovered that the rectified linear unit function (`relu`) is better for deep learning. (Unlike sigmoid, `relu` does not suffer from vanishing gradients.)
# 
# `Dropout` = dropout layer. These layers randomly eliminate some neurons to reduce overfitting. 
# 
# `Flatten` = input flattening layer. These layers take feature maps and transform them into 1D vectors.
# 
# `Dense` = fully connected layer. These are vanilla neural network layers, good for learning.
# 
# The `softmax` function normalizes the output to a probability distribution over predicted output classes.

# In[32]:


model = Sequential()

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape = (150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9))
model.add(Activation('softmax'))


# # Additional Parameters

# Neural networks are an optimization problem. We need to define:
# 1. a loss function (`categorical_crossentropy`)
# 2. an optimizer (`Adam`)
# 3. an evaluation metric (`accuracy`).

# In[33]:


model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy'])


# In[34]:


model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.samples // batch_size)*3,
        #steps_per_epoch=500,
        #epochs=50,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
        #validation_steps=500
)


# # Performance

# In[35]:


hist = model.history.history

import matplotlib.pyplot as plt

plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# From the Keras FAQ:
# 
# > Why is the training loss much higher than the testing loss?
# >
# >    A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.
# >
# >    Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.
# 

# # Confusion Matrix

# In[36]:


Y_pred = model.predict_generator(validation_generator, validation_generator.samples // batch_size+1)

Y_pred # distribution of softmax predictions over the predicted output classes


# In[37]:


y_pred = np.argmax(Y_pred, axis=1) # class predictions


# In[38]:


class_labels = list(train_generator.class_indices.keys())
class_labels


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))


# In[40]:


sum(sum(confusion_matrix(validation_generator.classes, y_pred))) == len(validation_generator.filepaths)


# We confirm that the number of items in the confusion matrix matches the size of the validation data.

# In[41]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

confusion_mtx = confusion_matrix(validation_generator.classes, y_pred) 

plot_confusion_matrix(confusion_mtx, classes = class_labels)


# # Classification Report

# In[42]:


print('Classification Report')

print(classification_report(validation_generator.classes, y_pred, target_names=class_labels))


# In[43]:


model.summary()


# # Filters

# In[44]:


layer_dict = dict([(layer.name, layer) for layer in model.layers])


# In[46]:


layer_name = 'conv2d_1'
filter_index = 0 # Which filter in this block would you like to visualise?

# Grab the filters and biases for that layer
filters, biases = layer_dict[layer_name].get_weights()

# Normalize filter values to a range of 0 to 1 so we can visualize them
f_min, f_max = np.amin(filters), np.amax(filters)
filters = (filters - f_min) / (f_max - f_min)

# Plot first few filters
n_filters, index = 6, 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    
    # Plot each channel separately
    for j in range(3):

        ax = plt.subplot(n_filters, 3, index)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.imshow(f[:, :, j], cmap='viridis')
        index += 1
        
plt.show()


# # Feature Maps

# In[47]:


from matplotlib import pyplot

for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)


# In[48]:


from keras.models import Model
Block1_Model = Model(inputs=model.inputs, outputs=model.layers[0].output)
Block2_Model = Model(inputs=model.inputs, outputs=model.layers[3].output)
Block3_Model = Model(inputs=model.inputs, outputs=model.layers[6].output)


# In[49]:


# load the image with the required shape
from keras.preprocessing.image import load_img, img_to_array

image = load_img(A[1], target_size=(150, 150))
pyplot.imshow(image)
pyplot.show()


# In[50]:


image = img_to_array(image)
image = image.reshape((1,) + image.shape)
image.shape


# ## Block 1

# In[51]:


import matplotlib.pyplot as plt

feature_maps = Block1_Model.predict(image)

square = 5
index = 1
for _ in range(square):
    for _ in range(square):
        
        ax = plt.subplot(square, square, index)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
        index += 1
        
plt.show()


# ## Block 2

# In[52]:


feature_maps = Block2_Model.predict(image)

square = 5
index = 1
for _ in range(square):
    for _ in range(square):
        
        ax = plt.subplot(square, square, index)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
        index += 1
        
plt.show()


# ## Block 3

# In[53]:


feature_maps = Block3_Model.predict(image)

square = 8
index = 1
for _ in range(square):
    for _ in range(square):
        
        ax = plt.subplot(square, square, index)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
        index += 1
        
plt.show()


# # Stress Test
# Given a random stock photo of an Erlenymeyer flask, can the classifier mark it as `Other`?

# In[54]:


image = load_img("/home/tim/Documents/Py/Flasks/Erlenmeyer_Flask.png", target_size=(150, 150))
pyplot.imshow(image)


# In[55]:


x = img_to_array(image)
x = x/255
x = x.reshape((1,) + x.shape)

import pandas as pd
df = pd.DataFrame(np.concatenate((np.expand_dims(class_labels, axis=1).T, (model.predict(x)).round(5)),axis=0))
print(df.to_string(index=False, header=False))


# The softmax function is almost completely mapped to the correct target.
# 
# Not only does the model correctly predict that the image belongs in the `Other` category, but it makes a **confident** prediction.

# In[56]:


image = load_img("/home/tim/Documents/Py/Flasks/StressTest/FlasksII/Part II/TypeA_179.jpg", target_size=(150, 150))
x = img_to_array(image)
pyplot.imshow(image)
x = x/255
x = x.reshape((1,) + x.shape)

import pandas as pd
df = pd.DataFrame(np.concatenate((np.expand_dims(class_labels, axis=1).T, (model.predict(x)).round(5)),axis=0))
print(df.to_string(index=False, header=False))


# However, the model incorrectly claims that an image of Flask A from a previous data set also belongs to the `Other` category.

# Conclusion: the model's predictions are closely coupled with the training environment. Need to vary the training environment for adequate generalization.

# # Acknowledgements
# 
# - Victor Luethen for providing the data
# - Swarnali Banerjee for helpful comments and critiques
# - Francois Chollet for authoring Keras and publishing several helpful tutorials
# - Jason Brownlee for publishing several helpful Keras tutorials on his website
# - Ryan Akilos for publishing GitHub code that was helpful for creating the confusion matrix
# - Shahariar Rabby for his Kaggle solution that was helpful for visualizing the confusion matrix
# - George Seif for his article concerning visualization of filters and feature maps
