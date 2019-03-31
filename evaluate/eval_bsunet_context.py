import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import nibabel as nib

from BSUnet import *
import BSUnetModified

global_path = "../data/Test"
global_weight_path = './weights/BottleNeckBSUnet/final_weights.hdf5'
global_autoencoder_path = './weights/AutoencoderBSUnet/after_epoch1.hdf5'

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
def preprocess(simg,sseg):
    simg[simg>250] = 250
    simg[simg<-200] = -200
    simg -= -200
    simg /= 450
    sseg[sseg>0] = 1
    return simg , sseg


def evaluate(model,fromIndex,batch_size=8,flag=False):
    path = global_path
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
            z = j - context
            data_point, data_point_seg = preprocess(img[:,:,j].astype(float),seg[:,:,j])
            data_point = data_point[...,np.newaxis]
            while z <= j+context:
                if z==j:
                    pass
                elif z < 0 :
                    simg , sseg = preprocess(img[:,:,0].astype(float),seg[:,:,0])
                    data_point = np.concatenate([data_point,simg[...,np.newaxis]],axis=2)
                elif z >= img.shape[2]:
                    simg , sseg = preprocess(img[:,:,img.shape[2]-1].astype(float),seg[:,:,img.shape[2]-1])
                    data_point = np.concatenate([data_point,simg[...,np.newaxis]],axis=2)
                else:
                    simg , sseg = preprocess(img[:,:,z].astype(float),seg[:,:,z])
                    data_point = np.concatenate([data_point,simg[...,np.newaxis]],axis=2)
                z += 1
            ## simg shape (512,512)
            ## treating tumor as part of liver
            if flag==False:
                if np.sum(data_point_seg == 1)>0 :
                    X_test.append(data_point)
                    y_test.append(data_point_seg)
            else:
                X_test.append(data_point)
                y_test.append(data_point_seg)
        print("Len of X_test ",len(X_test))
        X_test = np.array(X_test)
        y_test = np.array(y_test)
#         X_test = X_test[...,np.newaxis]
        y_test = y_test[...,np.newaxis]
        print("shape of X_train ",X_test.shape)
        print("Shape of y_train ",y_test.shape)
        history = model.evaluate(X_test,[y_test,np.zeros((y_test.shape[0],16,16,128))],batch_size=batch_size)
        print(history)
        histot.append(history)
    return histot

num_channels = 1
autoencoder_baseUnet = BSUnetModified.baseUNet(input_size=(512,512,num_channels),output_ch=(512,512,num_channels))
autoencoder_baseUnet.summary()

autoencoder_baseUnet.load_weights(global_autoencoder_path)

num_channels = 1
num_ct = 1
context = 1
# model = liverUnet(input_size=(512,512,num_channels))
# model = get_unet_sorr(input_size=(512,512,num_channels))
model = bottleneckFeatureUnet(input_size=(512,512,1+2*context),output_ch=(512,512,1))
# model_checkpoint = ModelCheckpoint('./weights/ContextBottleNeckUnet/best_weights.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.summary()
model.load_weights(global_weight_path)

h = evaluate(model,0,batch_size=2)

alps = np.array(h)
mean = np.mean(alps,axis=0)
print("Mean: "mean)
print(h)