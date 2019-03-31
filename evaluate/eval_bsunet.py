import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import nibabel as nib

from BSUnet import *

global_best_metric = 0
global_path = "../data/Test"
global_weight_path = './weights/BSUnet/final_weights.hdf5'
global_checkpoint_path = './weights/BSUnet/best_weights.hdf5'

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
            if flag==False:
                if np.sum(sseg == 1)>0 :
                    X_test.append(simg)
                    y_test.append(sseg)
            else:
                X_test.append(simg)
                y_test.append(sseg
        print("Len of X_test ",len(X_test))
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test = X_test[...,np.newaxis]
        y_test = y_test[...,np.newaxis]
        print("shape of X_train ",X_test.shape)
        print("Shape of y_train ",y_test.shape)
        history = model.evaluate(X_test,y_test,batch_size=batch_size)
        print(history)
        histot.append(history)
    return histot

num_channels = 1
num_ct = 1
# model = liverUnet(input_size=(512,512,num_channels))
# model = get_unet_sorr(input_size=(512,512,num_channels))
model = segmentedUnet(input_size=(512,512,num_channels),output_ch=(512,512,num_channels))
model_checkpoint = ModelCheckpoint(global_checkpoint_path, monitor='loss',verbose=1, save_best_only=True)
model.summary()

### Change the weights here ------------------------------------------------------------------------------------------------
model.load_weights(global_weight_path)
##--------------------------------------------------------------------------------------------------------------------------
h = evaluate(model,0,batch_size=2)

alp = np.array(h)
y = np.mean(alp,axis=0)
print("Metric : " ,y)     
print(h)                 
# values = {}
# histories = {}
# for e in range(12):
#     model.load_weights('./weights/BSUnet/after_epoch{}.hdf5'.format(e))
#     h = evaluate(model,0,batch_size = 2)
#     alp = np.array(h)
#     mean = np.mean(h,axis=0)
#     values[e] = mean
#     histories[e] = h