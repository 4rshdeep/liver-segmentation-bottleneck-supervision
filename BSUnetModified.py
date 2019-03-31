import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

    
def downBlock(input_img,input_channel,inner_channel,output_channel):

    tower_1 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(1,1))(input_img)

    tower_2 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(3,3))(input_img)

    tower_3 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(5,5))(input_img)

    output1 = concatenate([tower_1, tower_2, tower_3], axis=3)

    conv4 =  Conv2D(output_channel,kernel_size=(3,3),strides=(2,2),padding='same',activation='relu')(output1)

    conv5 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1))(conv4)

    out = BatchNormalization()(conv5)

    out = Activation(activation='relu')(out)

    return out

def denseBlock(input_img,input_channel,growth_rate,output_channel):

    conv1 = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1) , padding='same' )(input_img)
    b1 = BatchNormalization()(conv1)
    r1 = Activation(activation='relu')(b1)

    concat1 = concatenate([input_img,r1],axis=3)

    conv2 = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1), padding='same')(concat1)
    b2 = BatchNormalization()(conv2)
    r2 = Activation(activation='relu')(b2)

    concat2 = concatenate([concat1,r2],axis=3)
   


    return concat2


def upBlock(input_img,input_channel,inner_channel,output_channel,kernel_size):

    conv1 = Conv2D(inner_channel,kernel_size=(3,3),strides=(1,1),padding='same')(input_img)
    r1 = Activation(activation='relu')(conv1)

    conv2T = Conv2DTranspose(output_channel,kernel_size=kernel_size,strides=(2,2),padding='same')(r1)
    r2 = Activation(activation='relu')(conv2T)

    conv3 = Conv2D(output_channel,kernel_size=(3,3),strides=(1,1),padding='same')(r2)
    b = BatchNormalization()(conv3)
    r3 = Activation(activation='relu')(b)

    return r3

def transitionBlock(input_img,input_channel,output_channel):

    conv1 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)

    pool1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_img)
    conv2 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(pool1)

    conv31 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)
    conv32 = Conv2D(output_channel,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(conv31)

    conv41 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)
    conv42 = Conv2D(output_channel,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(conv41)

    concat = concatenate([conv1,conv2,conv32,conv42],axis=3)

    convf = Conv2D(output_channel,kernel_size=(3,3),strides=(1,1),padding='same')(concat)

    b = BatchNormalization()(convf)
    r = Activation(activation='relu')(b)

    return r

def baseUNet(input_size=(512,512,1),output_ch=(512,512,1)):
    inputs = Input(input_size)

    trans1 = transitionBlock(inputs,input_size[2],8)

    down1 = downBlock(trans1,input_channel=8,inner_channel=16,output_channel=16)

    down2 = downBlock(down1,input_channel=16,inner_channel=16,output_channel=32)

    down3 = downBlock(down2,input_channel=32,inner_channel=32,output_channel=64)

    down4 = downBlock(down3,input_channel=64,inner_channel=64,output_channel=96)

    down5 = downBlock(down4,input_channel=32,inner_channel=96,output_channel=128)

    trans2 = transitionBlock(down5,input_channel=128,output_channel=256)

    dense1 = denseBlock(trans2,input_channel=256,growth_rate=256,output_channel=128)
    
    conv3 = Conv2D(128,kernel_size=(3,3),strides=(1,1), padding='same',name="feature_vector")(dense1)
    b3 = BatchNormalization()(conv3)
    r3 = Activation(activation='relu')(b3)

    up1 = upBlock(r3,input_channel=128,inner_channel=128,output_channel=96,kernel_size=(2,2))

    up2 = upBlock(up1,input_channel=96,inner_channel=96,output_channel=64,kernel_size=(2,2))

    up3 = upBlock(up2,input_channel=64,inner_channel=64,output_channel=32,kernel_size=(2,2))

    up4 = upBlock(up3,input_channel=32,inner_channel=32,output_channel=16,kernel_size=(2,2))

    up5 = upBlock(up4,input_channel=16,inner_channel=16,output_channel=16,kernel_size=(2,2))

    convf = Conv2D(output_ch[2],kernel_size=(3,3),strides=(1,1),padding='same',activation='sigmoid',name="final_output")(up5)

    model = Model(inputs=[inputs], outputs=[convf,conv3])
    
    losses = {
    "final_output": "binary_crossentropy",
    "feature_vector": "categorical_crossentropy",
    }
    lossWeights = {"final_output": 1.0, "feature_vector": 0.0}

    model.compile(optimizer=Adam(lr=1e-3), loss=losses,loss_weights=lossWeights, metrics=['accuracy'])
    
    return model

def segmentedUnet(input_size=(512,512,1),output_ch=(512,512,1)):
    inputs = Input(input_size)

    trans1 = transitionBlock(inputs,input_size[2],8)

    down1 = downBlock(trans1,input_channel=8,inner_channel=16,output_channel=16)

    down2 = downBlock(down1,input_channel=16,inner_channel=16,output_channel=32)

    down3 = downBlock(down2,input_channel=32,inner_channel=32,output_channel=64)

    down4 = downBlock(down3,input_channel=64,inner_channel=64,output_channel=96)

    down5 = downBlock(down4,input_channel=32,inner_channel=96,output_channel=128)

    trans2 = transitionBlock(down5,input_channel=128,output_channel=256)

    dense1 = denseBlock(trans2,input_channel=256,growth_rate=256,output_channel=128)

    concat1 = concatenate([dense1,down5],axis=3)

    up1 = upBlock(concat1,input_channel=128,inner_channel=128,output_channel=96,kernel_size=(2,2))

    concat2 = concatenate([up1,down4],axis=3)
    up2 = upBlock(concat2,input_channel=96,inner_channel=96,output_channel=64,kernel_size=(2,2))

    concat3 = concatenate([up2,down3],axis=3)
    up3 = upBlock(concat3,input_channel=64,inner_channel=64,output_channel=32,kernel_size=(2,2))

    concat4 = concatenate([up3,down2],axis=3)
    up4 = upBlock(concat4,input_channel=32,inner_channel=32,output_channel=16,kernel_size=(2,2))

    concat5 = concatenate([up4,down1],axis=3)
    up5 = upBlock(concat5,input_channel=16,inner_channel=16,output_channel=16,kernel_size=(2,2))

    convf = Conv2D(output_ch[2],kernel_size=(1,1),padding='same',activation='sigmoid')(up5)

    model = Model(inputs=[inputs], outputs=[convf])

    model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_coef])
    
    return model