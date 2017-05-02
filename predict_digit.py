#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
#import numpy as np
#from matplotlib import image as mpimg
#from matplotlib import pyplot as plt

import keras
from keras.layers.core import *
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import *
from PIL import Image

import sys

img_rows, img_cols = 28, 28
input_shape=(img_rows, img_cols,1)
def get_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3,activation='relu',
   input_shape=input_shape ))
	model.add(Convolution2D(64,3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	#model.add(Dense(128, activation='relu'))
	#model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(Adam(),"categorical_crossentropy",metrics=['accuracy'])

	model.load_weights("model0.h5")
	return model
model=get_model()

def preprocess_img(img_name, out_shape=(28,28)):
    img_orig=Image.open(img_name)
    img = img_orig.convert('L')
    img=img.resize(out_shape, Image.ANTIALIAS)
    img=np.array(img)
    img=img.astype('float32')
    img/=255
    img=1-img
    img_np=np.expand_dims(img,-1)
    return img_np

def predict_drawn_img(img_name):
    global model
    img=preprocess_img(img_name)
    inp=np.expand_dims(img,0)
    pred=model.predict(inp)
    ans=np.argmax(pred,1)
    return ans

if __name__=="__main__":
    fname=sys.argv[1]
    print(predict_drawn_img(fname)[0])
