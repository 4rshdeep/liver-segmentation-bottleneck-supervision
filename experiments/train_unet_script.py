import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import nibabel as nib

from liverModel import *
from liverData import *

num_of_epochs = 1
global_best_metric = 0


def read_ct(path):
    img = nib.load(path)
    img = img.get_data()
    return img
def loadCT(path):
    images = glob.glob(path+"/volume*")
    segmentations = glob.glob(path+"/segmentation*", )
    images = sorted(images)
    segmentations = sorted(segmentations)
    return images , segmentations
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def trainUnet(model,model_checkpoint,num_channels=2,num_ct=1,folders=2,batch_size=8):
    """
        Training by taking ct scans of only num_ct files and each data point of shape
        (512,512,num_channels)
    """
    path = "../data/batch"
    images ,segmentations = loadCT(path)
    for i in range(0,len(images),num_ct):
        print("image " + str(i)+" out of "+str(len(images)))
        X_train = []
        y_train = []
        img = read_ct(images[i])
        seg = read_ct(segmentations[i])
        print("Shape of img : ", img.shape)
        ##img shape: (512,512,X) X is the sum of all slices of num_ct files
        for j in range(0,img.shape[2]):
            ## simg shape (512,512)
            simg = img[:,:,j].astype(float)
            sseg = seg[:,:,j]
            ##HU clipping
            simg[simg >250 ] = 250
            simg[simg < -200] = -200
            ## Normalization
            simg -= -200
            simg /= 450
            ## treating tumor as part of liver
            sseg[sseg > 0] = 1
            if np.sum(sseg == 1)>0 :
                X_train.append(simg)
                y_train.append(sseg)
        print("Len of X_train ",len(X_train))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = X_train[...,np.newaxis]
        y_train = y_train[...,np.newaxis]
#         mean = np.mean(X_train)  # mean for data centering
#         std = np.std(X_train)  # std for data normalization

#         X_train -= mean
#         X_train /= std
        print("shape of X_train ",X_train.shape)
        print("Shape of y_train ",y_train.shape)
        model.fit(X_train,y_train,callbacks=[model_checkpoint],batch_size=batch_size) ## set epoch to 1
    return model

def evaluate(model,fromIndex,batch_size=8):
    path = "../data/Test"
    images ,segmentations = loadCT(path)
    histot = []
    for i in range(fromIndex,len(images)):
        print("image " + str(i))
        X_test = []
        y_test = []
        img = read_ct(images[i])
        seg = read_ct(segmentations[i])
        print("Shape of img : ", img.shape)
        ##img shape: (512,512,X) X is the sum of all slices of num_ct files
        for j in range(0,img.shape[2]):
            simg = img[:,:,j].astype(float)
            sseg = seg[:,:,j]
            ## simg shape (512,512)
            ##HU clipping
            simg[simg >250 ] = 250
            simg[simg < -200] = -200
            ## Normalization
            simg -= -200
            simg /= 450
            ## treating tumor as part of liver
            sseg[sseg > 0] = 1
            if np.sum(sseg == 1)>0 :
                X_test.append(simg)
                y_test.append(sseg)
        print("Len of X_test ",len(X_test))
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test = X_test[...,np.newaxis]
        y_test = y_test[...,np.newaxis]
#         mean = np.mean(X_test)  # mean for data centering
#         std = np.std(X_test)  # std for data normalization

#         X_test -= mean
#         X_test /= std
        print("shape of X_train ",X_test.shape)
        print("Shape of y_train ",y_test.shape)
        history = model.evaluate(X_test,y_test,batch_size=batch_size)
        print(history)
        histot.append(history)
    return histot

####################################################################################################################################################

num_channels = 1
num_ct = 1
# model = liverUnet(input_size=(512,512,num_channels))
model = get_unet_sorr(input_size=(512,512,num_channels))
model_checkpoint = ModelCheckpoint('weights/best_weights.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.summary()
model.load_weights('weights/final_weights.hdf5')

num_epochs = 20
for e in range(num_epochs):
    print("*"*50)
    print("** epoch ",e)
    print("*"*50)
    model = trainUnet(model,model_checkpoint,num_channels=num_channels,num_ct=num_ct,folders=1,batch_size=10)
    model.save_weights('weights/after_epoch{}.hdf5'.format(e))
    model.save_weights('weights/final_weights.hdf5'.format(e))


h = evaluate(model,0,batch_size=2) 

alp = np.array(h)
y = np.mean(alp,axis=0)
print("Average Loss and Average dice coefficient : ", y)

# print(global_best_metric)
# if y[1]>global_best_metric[1]:
#     global_best_metric = y