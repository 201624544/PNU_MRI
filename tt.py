import os
import json
import glob
import random
import collections

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("/home/bono/rsna/train_labels.csv")
sample_df = pd.read_csv("/home/bono/rsna/sample_submission.csv")

import os
for dirname, _, filenames in os.walk('/home/bono/rsna/train'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('/home/bono/rsna/test'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow_hub as hub
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_dicom(path):
    dicom=pydicom.read_file(path)
    data=dicom.pixel_array
    data=data-np.min(data)
    if np.max(data) != 0:
        data=data/np.max(data)
    data=(data*255).astype(np.uint8)
    return data

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

train_dir='/home/bono/rsna/train'
trainset=[]
trainlabel=[]
trainidt=[]
for i in tqdm(range(len(train_df))):
    idt=train_df.loc[i,'BraTS21ID']
    idt2=('00000'+str(idt))[-5:]
    path=os.path.join(train_dir,idt2,'T1wCE')              
    for im in os.listdir(path):
        img=load_dicom(os.path.join(path,im)) 
        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0
        trainset+=[image]
        trainlabel+=[train_df.loc[i,'MGMT_value']]
        trainidt+=[idt]

test_dir='/home/bono/rsna/test'
testset=[]
testidt=[]
for i in tqdm(range(len(sample_df))):
    idt=sample_df.loc[i,'BraTS21ID']
    idt2=('00000'+str(idt))[-5:]
    path=os.path.join(test_dir,idt2,'T1wCE')               
    for im in os.listdir(path):   
        img=load_dicom(os.path.join(path,im))
        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0
        testset+=[image]
        testidt+=[idt]

y=np.array(trainlabel)
Y_train=to_categorical(y)
X_train=np.array(trainset)
X_test=np.array(testset)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64,kernel_size=(4,4),input_shape=(64,64,1),activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal"))
model.add(keras.layers.Dense(2,"softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer = "RMSprop", metrics=["accuracy"])

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=8)

hist = model.fit(X_train, Y_train, epochs=10, batch_size=64, verbose=1,callbacks=[callback])

get_ac = hist.history['acc']
get_los = hist.history['loss']

print("---------------------------------------")

epochs = range(len(get_ac))
plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)
plt.figure()
plt.show()
plt.savefig('tt.png')

y_pred=model.predict(X_test)
pred=np.argmax(y_pred,axis=1)
result=pd.DataFrame(testidt)
result[1]=pred
result.columns=['BraTS21ID','MGMT_value']
result2=result.groupby('BraTS21ID',as_index=False).mean()
result2

print(result2)
print("---------------------------------------")

result3 = result2.sort_values(by='MGMT_value', ascending=False)
result3

result3['BraTS21ID']=sample_df['BraTS21ID']
result3['MGMT_value']=result3['MGMT_value'].apply(lambda x:round(x*10)/10)
result3.to_csv('submission.csv',index=False)
result3
print(result3)
