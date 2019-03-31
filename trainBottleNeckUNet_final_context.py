
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import nibabel as nib


# In[4]:


from BSUnet import *
import BSUnetModified


# In[5]:


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
def preprocess(simg,sseg):
    simg[simg>250] = 250
    simg[simg<-200] = -200
    simg -= -200
    simg /= 450
    sseg[sseg>0] = 1
    return simg , sseg


# In[13]:


def trainBottleNeckUnet(model,model_checkpoint,autoencoder_model,num_channels=2,num_ct=1,folders=2,batch_size=8,context=2):
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
            if np.sum(data_point_seg == 1)>0 :
                X_train.append(data_point)
                y_train.append(data_point_seg)
        print("Len of X_train ",len(X_train))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
#         X_train = X_train[...,np.newaxis]
        y_train = y_train[...,np.newaxis]
        print("shape of X_train ",X_train.shape)
        print("Shape of y_train ",y_train.shape)
        
#         for k in range(0,X_train.shape[0],batch_size):
        output1 = autoencoder_model.predict(y_train)
        feature_vectors_autoencoder = output1[1]
        model.fit(X_train,[y_train,feature_vectors_autoencoder],callbacks=[model_checkpoint],batch_size=batch_size) ## set epoch to 1
    return model


# In[7]:


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
        history = model.evaluate(X_test,[y_test,np.zeros((y_test.shape[0],16,16,128))],batch_size=batch_size)
        print(history)
        histot.append(history)
    return histot


# In[8]:


num_channels = 1
autoencoder_baseUnet = BSUnetModified.baseUNet(input_size=(512,512,num_channels),output_ch=(512,512,num_channels))
autoencoder_baseUnet.summary()


# In[9]:


autoencoder_baseUnet.load_weights('./weights/AutoencoderBSUnet/after_epoch1.hdf5')


# In[15]:


num_channels = 1
num_ct = 1
context = 1
# model = liverUnet(input_size=(512,512,num_channels))
# model = get_unet_sorr(input_size=(512,512,num_channels))
model = bottleneckFeatureUnet(input_size=(512,512,1+2*context),output_ch=(512,512,1))
model_checkpoint = ModelCheckpoint('./weights/ContextBottleNeckUnet/best_weights.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.summary()
model.load_weights('./weights/ContextBottleNeckUnet/final_weights.hdf5')


# In[16]:


num_epochs = 2
for e in range(num_epochs):
    print("*"*50)
    print("** epoch ",e)
    model = trainBottleNeckUnet(model,model_checkpoint,autoencoder_baseUnet,num_channels=num_channels,num_ct=num_ct,folders=1,
                                batch_size=10,context=1)
    model.save_weights('./weights/ContextBottleNeckUnet/after_epoch{}.hdf5'.format(e+3))
    model.save_weights('./weights/ContextBottleNeckUnet/final_weights.hdf5')


# # Validation

# In[ ]:


# h = evaluate(model,0,batch_size=2)


# In[ ]:


# alps = np.array(h)
# mean = np.mean(alps,axis=0)
# print(mean[3])

