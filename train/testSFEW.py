# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten, Dense, Input , MaxPooling2D, Convolution2D, Activation, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from tqdm import tqdm
from keras import backend as K
import pickle as pkl
import glob

def getModel(fc_secondlast,fc_last):
    #VGGbottomMODEL################################################################################################     
    img_input = Input(shape=(224, 224, 3))
    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    model0 = Model(img_input, x)    
    
    img_input1 = Input(shape=(224, 224, 3))
    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_11')(img_input1)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_21')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool11')(x)
    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_11')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_21')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool21')(x)
    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_11')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_21')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_31')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool31')(x)
    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_11')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_21')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_31')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool41')(x)
    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_11')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_21')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_31')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool51')(x)
    model1 = Model(img_input1, x)           
    
    img_input2 = Input(shape=(224, 224, 3))
    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_12')(img_input2)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_22')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool12')(x)
    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_12')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_22')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool22')(x)
    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_12')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_22')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_32')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool32')(x)
    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_12')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_22')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_32')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool42')(x)
    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_12')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_22')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_32')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool52')(x)
    model2 = Model(img_input2, x)     
    
    #VGGbottomMODEL################################################################################################    
    
    eye_input  = Input(shape=(224, 224, 3))
    nos_input  = Input(shape=(224, 224, 3))
    mou_input  = Input(shape=(224, 224, 3))
    eye  = model0(eye_input)
    nose = model1(nos_input)
    mouth= model2(mou_input)
    
    branch1 = Flatten(name='flatten6-1')(eye)
    branch1 = Dense(1024, name='fc6-1', kernel_initializer='glorot_normal')(branch1)
    branch1 = Dropout(0.5)(branch1)
    branch1 = Activation('relu', name='fc6-1/relu')(branch1)        
    branch2 = Flatten(name='flatten6-2')(nose)
    branch2 = Dense(1024, name='fc6-2', kernel_initializer='glorot_normal')(branch2)
    branch2 = Dropout(0.5)(branch2)
    branch2 = Activation('relu', name='fc6-2/relu')(branch2)        
    branch3 = Flatten(name='flatten6-3')(mouth)
    branch3 = Dense(1024, name='fc6-3', kernel_initializer='glorot_normal')(branch3)
    branch3 = Dropout(0.5)(branch3)
    branch3 = Activation('relu', name='fc6-3/relu')(branch3)
    
    x = Concatenate(axis=1)([branch1,branch2,branch3])
    x = Dense(fc_secondlast, name='fc7', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu', name='fc7/relu')(x)
    if fc_last!=0:
        x = Dense(fc_last, name='fc8', kernel_initializer='glorot_normal')(x)
        x = Dropout(0.5)(x)
        x = Activation('relu', name='fc8/relu')(x)
    x = Dense(nb_class, name='fc9', kernel_initializer='glorot_normal')(x)
    x = Activation('softmax', name='fc9/softmax')(x)         
    emotion_model = Model([eye_input,nos_input,mou_input], x)

    return emotion_model

def getVGGModel(fc_secondlast,fc_last):
#VGGbottomMODEL################################################################################################     
    img_input = Input(shape=(224, 224, 3))
    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    model = Model(img_input, x)   
    pretrained_vgg ='./weight/vggface_tf_notop.h5'
    model.load_weights(pretrained_vgg)
    
    last_layer = model.get_layer('pool5').output       
    x = Flatten(name='flatten')(last_layer)
    x = Dense(fc_secondlast, name='fc6', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu', name='fc6/relu')(x)
    if fc_last!=0:
        x = Dense(fc_last, name='fc7', kernel_initializer='glorot_normal')(x)
        x = Dropout(0.5)(x)
        x = Activation('relu', name='fc7/relu')(x)
    x = Dense(nb_class, name='fc8', kernel_initializer='glorot_normal')(x)
    x = Activation('softmax', name='fc8/softmax')(x)     
    emotion_model = Model(model.input, x)
    emotion_model.summary()
    return emotion_model
nb_class = 7    
if os.path.isfile('dataset.pkl'):
    [fg_images,fg_labels]=pkl.load(open('dataset.pkl'))

else:
    from util.tools import extractAlignedFace
    from util.detector import detectFace
    
    SFEW_dataroot = './data/SFEW'
    CATE =      ['Angry','Neutral','Disgust'   ,'Fear'  ,'Happy'  ,'Sad','Surprise']
    SFEW_CATE=  ['Angry','Disgust','Fear'      ,'Happy' ,'Neutral','Sad','Surprise']
    source_images_test   = glob.glob(os.path.join(SFEW_dataroot,'Test', '*.*'))
    source_images_test.sort()
    ans = open(os.path.join(SFEW_dataroot, 'assume_ans'))
    ans = ans.readlines()
    align = True
    finegrain = True
    
    images,labels=[],[]
    for imgpath in tqdm(source_images_test):
        X  = image.load_img( imgpath ) 
        images.append( X )
        for info in ans:
            if imgpath.find(info.split('\t')[0]) !=-1:
                sfew_label = SFEW_CATE[int(info.split('\t')[1].split('\n')[0])]
                labels.append( sfew_label  )
                break
            
     
           
    fg_images=[]
    ful_images=[]
    fg_labels=[]
    threshold = [0.5,0.5,0.6]  
    for img,lbl in tqdm(zip(images,labels))    :
    
    
        X  = image.img_to_array(img)
        threshold = [0.6,0.6,0.7] 
        rec = detectFace(X,threshold)
        rec = np.array(rec)    
        #print('1')
        facefind = True
        if len(rec)!=0:
            rec=rec[rec[:,4]==np.max(rec[:,4])][0]
            X, face_rec = extractAlignedFace(X[:,:,::-1],rec[5:15].reshape(5,2))
        else:         
            #print('2')
            threshold = [0.5,0.5,0.6] 
            rec = detectFace(X,threshold)
            rec = np.array(rec)        
            if len(rec)!=0:
                rec=rec[rec[:,4]==np.max(rec[:,4])][0]
                X, face_rec = extractAlignedFace(X[:,:,::-1],rec[5:15].reshape(5,2))       
            else:     
                #print('3')        
                threshold = [0.4,0.4,0.5] 
                rec = detectFace(X,threshold)
                rec = np.array(rec)        
                if len(rec)!=0:
                    rec=rec[rec[:,4]==np.max(rec[:,4])][0]
                    X, face_rec = extractAlignedFace(X[:,:,::-1],rec[5:15].reshape(5,2))       
                else: 
                    #print('4')
                    allrec=[]
                    allimg=[]
                    threshold = [0.4,0.4,0] 
                    detectScale =[2,1.7,1.5,1,0.7,0.5]
                    rotation = [-20,-15,-12,-10,0,10,12,15,20]
                    
                    for s in detectScale:
                        dec_img = cv2.resize( X ,None, fx=s,fy=s)
                        for r in rotation:
                            (col,row,channel) = dec_img.shape
                            M = cv2.getRotationMatrix2D((col,row),-10,1)
                            dec_img = cv2.warpAffine(dec_img,M,(col,row))
                        
                            rec = detectFace(dec_img,threshold)
                            rec = np.array(rec)
                            if len(rec)!=0:
                                rec=rec[rec[:,4]==np.max(rec[:,4])][0]   
                                X, face_rec = extractAlignedFace(dec_img[:,:,::-1],rec[5:15].reshape(5,2))
                                allrec.append(rec) 
                                allimg.append(X)
                                #print rec[4]
                    if len(allrec)!=0:
                        allrec=np.array(allrec)
                        recmax=np.where(allrec[:,4]==np.max(allrec[:,4]))[0][0]
                        allimg=np.array(allimg)
                        X = allimg[recmax]                  
            
                    else:
                        facefind = False
    
    
    
        if facefind:
            X = np.expand_dims(X, axis=0) 
            X = preprocess_input(X)
            X = np.array(X)        
            
            eyey1= face_rec[0,1]
            eyey2= face_rec[1,1]
            nosy = face_rec[2,1]
            mouy1= face_rec[3,1]
            mouy2= face_rec[4,1]
            
            y1 = int(np.mean((eyey1,eyey2)))
            y2 = int(nosy)
            y3 = int(np.mean((mouy1,mouy2)))
            X1 = X[0,:][0:y2][:]
            X2 = X[0,:][y1:y3][:]
            X3 = X[0,:][y2::][:]
            X1 = cv2.resize(X1,(224,224))
            X2 = cv2.resize(X2,(224,224))
            X3 = cv2.resize(X3,(224,224))
            ful_images.append(X[0])
            fg_images.append([X1,X2,X3])
        else:
            X = np.zeros((224,224,3))
            X = np.array(X)   
            ful_images.append(X)
            fg_images.append([X,X,X])
        fg_labels.append( CATE.index(lbl) )  
    ful_images= np.array(ful_images)    
    fg_images = np.array(fg_images)
    fg_labels = np_utils.to_categorical(fg_labels,nb_class)
    pkl.dump([fg_images,fg_labels],open('dataset.pkl','w'))







#EVALUATE################################################################################################      

fg_inimg =[fg_images[:,0,:,:],fg_images[:,1,:,:],fg_images[:,2,:,:]]
gt = np.argmax(fg_labels,axis=1)

emov4_0= getModel(2048,0)
emov4_1= getModel(2048,1024)
emov4_2= getModel(3072,0)
emov4_3= getModel(3072,2048)
emov4_4= getModel(3072,3072) 
emov4_5= getModel(4096,0)
emov4_6= getModel(4096,3072)
emov4_7= getModel(4096,4096)
answers =[]
accuracy=[]
for i in tqdm(range(8)):
    eval('emov4_{}.load_weights(\'{}\')'.format(i,glob.glob('./weight/result/*_{}_weights*'.format(i))[0]))
    tmp = eval('emov4_{}.predict(fg_inimg)'.format(i,i)) 
    answers.append(tmp)
    accuracy.append( np.count_nonzero(  np.argmax(tmp,axis=1) == gt) / 372.0 ) 
pkl.dump([answers,accuracy],open('ans0_7','w'))

emov4_8= getModel(2048,0)
emov4_9= getModel(2048,1024)
emov4_10= getModel(3072,0)
emov4_11= getModel(3072,2048)
emov4_12= getModel(3072,3072) 
emov4_13= getModel(4096,0)
emov4_14= getModel(4096,3072)
emov4_15= getModel(4096,4096)
answers =[]
accuracy=[]
for i in tqdm(range(8,16)):
    eval('emov4_{}.load_weights(\'{}\')'.format(i,glob.glob('./weight/result/*_{}_weights*'.format(i))[0]))
    tmp = eval('emov4_{}.predict(fg_inimg)'.format(i,i)) 
    answers.append(tmp)
    accuracy.append( np.count_nonzero(  np.argmax(tmp,axis=1) == gt) / 372.0 ) 
pkl.dump([answers,accuracy],open('ans8_15','w'))

emov4_16= getModel(2048,0)
emov4_17= getModel(2048,1024)
emov4_18= getModel(3072,0)
emov4_19= getModel(3072,2048)
emov4_20= getModel(3072,3072) 
emov4_21= getModel(4096,0)
emov4_22= getModel(4096,3072)
emov4_23= getModel(4096,4096)
answers =[]
accuracy=[]
for i in tqdm(range(16,24)):
    eval('emov4_{}.load_weights(\'{}\')'.format(i,glob.glob('./weight/result/*_{}_weights*'.format(i))[0]))
    tmp = eval('emov4_{}.predict(fg_inimg)'.format(i,i)) 
    answers.append(tmp)
    accuracy.append( np.count_nonzero(  np.argmax(tmp,axis=1) == gt) / 372.0 ) 
pkl.dump([answers,accuracy],open('ans16_23','w'))

emov4_24= getModel(2048,0)
emov4_25= getModel(2048,1024)
emov4_26= getModel(3072,0)
emov4_27= getModel(3072,2048)
emov4_28= getModel(3072,3072) 
emov4_29= getModel(4096,0)
emov4_30= getModel(4096,3072)
emov4_31= getModel(4096,4096)
answers =[]
accuracy=[]
for i in tqdm(range(24,32)):
    eval('emov4_{}.load_weights(\'{}\')'.format(i,glob.glob('./weight/result/*_{}_weights*'.format(i))[0]))
    tmp = eval('emov4_{}.predict(fg_inimg)'.format(i,i)) 
    answers.append(tmp)
    accuracy.append( np.count_nonzero(  np.argmax(tmp,axis=1) == gt) / 372.0 ) 
pkl.dump([answers,accuracy],open('ans24_31','w'))


emov4_32= getVGGModel(4096,4096)
emov4_32.load_weights('./weight/result/emov4_32_weights.119-6.69.hdf5')
vgg_ans = emov4_32.predict(ful_images)
vgg_acc = np.count_nonzero(  np.argmax(vgg_ans,axis=1) == gt) / 372.0






[ans0,acc0] = pkl.load(open('ans0_7'))
[ans8,acc8] = pkl.load(open('ans8_15'))
[ans16,acc16] = pkl.load(open('ans16_23'))
[ans24,acc24] = pkl.load(open('ans24_31'))

fus1  = ans0[0] + ans0[1] + ans0[4] + ans0[5] + ans0[7] +  ans24[0] + ans24[1] + ans24[4] + ans24[5] + ans24[7]
acc_fus1=  np.count_nonzero(  np.argmax(fus1,axis=1) == gt) / 372.0
fus1_2= ans8[0] + ans8[1] + ans8[4] + ans8[5] + ans8[7] +  ans16[0] + ans16[1] + ans16[4] + ans16[5] + ans16[7]
acc_fus1_2= np.count_nonzero(  np.argmax(fus1_2,axis=1) == gt) / 372.0
fus2  = ans0[0] + ans0[2] + ans0[3] + ans0[4] + ans0[7] +  ans24[0] + ans24[2] + ans24[3] + ans24[4] + ans24[7]
acc_fus2 =  np.count_nonzero(  np.argmax(fus2 ,axis=1) == gt) / 372.0
fus2_2= ans8[0] + ans8[2] + ans8[3] + ans8[4] + ans8[7] +  ans16[0] + ans16[2] + ans16[3] + ans16[4] + ans16[7]
acc_fus2_2=  np.count_nonzero(  np.argmax(fus2_2,axis=1) == gt) / 372.0

fus_all = np.zeros((372, 7))
for i in range(8):
    fus_all+= ans0[i]+ans8[i]+ans16[i]+ans24[i]
acc_fus_all=  np.count_nonzero(  np.argmax(fus_all,axis=1) == gt) / 372.0


fus_all_noRAFD = np.zeros((372, 7))
for i in range(8):
    fus_all_noRAFD+= ans0[i]+ans24[i]
acc_fus_all_noRAFD=  np.count_nonzero(  np.argmax(fus_all_noRAFD,axis=1) == gt) / 372.0


#emotion_model2_0 = getModel(2048,1024)
#emotion_model2_1 = getModel(3072,2048)
#emotion_model2_2 = getModel(4096,3072)
#emotion_model2_3 = getModel(4096,0)
#emotion_model2_4 = getModel(2048,0)
#emotion_model2_5 = getModel(3072,0)
#emotion_model2_6 = getModel(4096,4096)
#emotion_model2_7 = getModel(3072,3072)
#emotion_model2_16 = getModel(3072,0)
#emotion_model2_17 = getModel(3072,2048)
#emotion_model2_18 = getModel(3072,3072)    
#emotion_model2_19 = getModel(4096,0)
#emotion_model2_20 = getModel(4096,3072)
#emotion_model2_21 = getModel(4096,4096) 
#emotion_model2_22 = getModel(2048,0)
#emotion_model2_23 = getModel(2048,1024)
#emotion_model2_0.load_weights('./weight/train_result/emov2_0_weights.69-3.39.hdf5')    
#emotion_model2_1.load_weights('./weight/train_result/emov2_1_weights.99-3.75.hdf5')
#emotion_model2_2.load_weights('./weight/train_result/emov2_2_weights.69-3.60.hdf5')
#emotion_model2_3.load_weights('./weight/train_result/emov2_3_weights.99-3.52.hdf5')    
#emotion_model2_4.load_weights('./weight/train_result/emov2_4_weights.99-3.63.hdf5')    
#emotion_model2_5.load_weights('./weight/train_result/emov2_5_weights.99-3.18.hdf5')  
#emotion_model2_6.load_weights('./weight/train_result/emov2_6_weights.119-3.82.hdf5')    
#emotion_model2_7.load_weights('./weight/train_result/emov2_7_weights.99-3.64.hdf5')
#emotion_model2_22.load_weights('./weight/train_result/emov2_22_weights.59-3.18.hdf5')
#emotion_model2_23.load_weights('./weight/train_result/emov2_23_weights.89-4.02.hdf5')
#emotion_model2_16.load_weights('./weight/train_result/emov2_16_weights.89-3.82.hdf5')
#emotion_model2_17.load_weights('./weight/train_result/emov2_17_weights.129-3.82.hdf5')
#emotion_model2_18.load_weights('./weight/train_result/emov2_18_weights.159-4.84.hdf5')
#emotion_model2_19.load_weights('./weight/train_result/emov2_19_weights.69-3.55.hdf5')
#emotion_model2_20.load_weights('./weight/train_result/emov2_20_weights.149-3.38.hdf5')
#emotion_model2_21.load_weights('./weight/train_result/emov2_21_weights.139-3.59.hdf5')
#ans0 = emotion_model2_0.predict(fg_inimg)
#ans1 = emotion_model2_1.predict(fg_inimg)
#ans2 = emotion_model2_2.predict(fg_inimg)
#ans3 = emotion_model2_3.predict(fg_inimg)
#ans4 = emotion_model2_4.predict(fg_inimg)
#ans5 = emotion_model2_5.predict(fg_inimg)
#ans6 = emotion_model2_6.predict(fg_inimg)
#ans7 = emotion_model2_7.predict(fg_inimg)
#ans22 = emotion_model2_22.predict(fg_inimg)
#ans23 = emotion_model2_23.predict(fg_inimg)
#ans16 = emotion_model2_16.predict(fg_inimg)
#ans17 = emotion_model2_17.predict(fg_inimg)
#ans18 = emotion_model2_18.predict(fg_inimg)
#ans19 = emotion_model2_19.predict(fg_inimg)
#ans20 = emotion_model2_20.predict(fg_inimg)
#ans21 = emotion_model2_21.predict(fg_inimg)
#acc0 = np.count_nonzero(  np.argmax(ans0,axis=1) == gt) / 372.0
#acc1 = np.count_nonzero(  np.argmax(ans1,axis=1) == gt) / 372.0
#acc2 = np.count_nonzero(  np.argmax(ans2,axis=1) == gt) / 372.0
#acc3 = np.count_nonzero(  np.argmax(ans3,axis=1) == gt) / 372.0
#acc4 = np.count_nonzero(  np.argmax(ans4,axis=1) == gt) / 372.0
#acc5 = np.count_nonzero(  np.argmax(ans5,axis=1) == gt) / 372.0
#acc6 = np.count_nonzero(  np.argmax(ans6,axis=1) == gt) / 372.0
#acc7 = np.count_nonzero(  np.argmax(ans7,axis=1) == gt) / 372.0
#acc22 = np.count_nonzero(  np.argmax(ans22,axis=1) == gt) / 372.0
#acc23 = np.count_nonzero(  np.argmax(ans23,axis=1) == gt) / 372.0
#acc16 = np.count_nonzero(  np.argmax(ans16,axis=1) == gt) / 372.0
#acc17 = np.count_nonzero(  np.argmax(ans17,axis=1) == gt) / 372.0
#acc18 = np.count_nonzero(  np.argmax(ans18,axis=1) == gt) / 372.0
#acc19 = np.count_nonzero(  np.argmax(ans19,axis=1) == gt) / 372.0
#acc20 = np.count_nonzero(  np.argmax(ans20,axis=1) == gt) / 372.0
#acc21 = np.count_nonzero(  np.argmax(ans21,axis=1) == gt) / 372.0
#import pickle as pkl
#pkl.dump([ans0,ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans16,ans17,ans18,ans19,ans20,ans21,ans22,ans23], open( "tmp.p", "w" ) )
#fullans1 = ans0+ans1+ans4+ans5+ans7
#fullans2 = ans0+ans1+ans3+ans4+ans5+ans6+ans7
#fullans3 = ans17+ans18+ans20+ans21+ans23
#fullans4 = ans16+ans17+ans18+ans19+ans20+ans21+ans23
#fullans5 = fullans1+fullans3
#fullans6 = fullans2+fullans4
#fullans7 = fullans6 + ans2 + ans22
#fullans8 = ans0+ans1+ans4+ans6+ans7
#fullans9 = 
#facc1 = np.count_nonzero(  np.argmax(fullans1,axis=1) == gt) / 372.0
#facc2 = np.count_nonzero(  np.argmax(fullans2,axis=1) == gt) / 372.0    
#facc3 = np.count_nonzero(  np.argmax(fullans3,axis=1) == gt) / 372.0
#facc4 = np.count_nonzero(  np.argmax(fullans4,axis=1) == gt) / 372.0
#facc5 = np.count_nonzero(  np.argmax(fullans5,axis=1) == gt) / 372.0
#facc6 = np.count_nonzero(  np.argmax(fullans6,axis=1) == gt) / 372.0
#facc7 = np.count_nonzero(  np.argmax(fullans7,axis=1) == gt) / 372.0
#facc8 = np.count_nonzero(  np.argmax(fullans8,axis=1) == gt) / 372.0
#
#
#F = []
#A = []
#for i in tqdm(range(np.power(2,16))):
#    sw =  bin(i)[2::].zfill(16)
#    fuans=np.zeros((372,7))
#    for j in  range(16):
#        if j<=7:
#            index = j
#        else:
#            index = j+8
#            
#        if sw[j]=='1':
#            fuans+= eval('ans{}'.format(index))
#        elif sw[j] =='0':
#            tmp=0
#        else:
#            print 'GG'
#
#    fuacc = np.count_nonzero(  np.argmax(fuans,axis=1) == gt) / 372.0 
#    print fuacc 
#    F.append(fuacc)
#    A.append(np.argmax(fuans,axis=1) )
#
#
#def confus_mtx(given_answer, pred):
#    cfs_mtx = np.zeros((7,7))
#    for a, p in zip(given_answer, pred):
#        p=int(p)
#        if a==p:
#            cfs_mtx[a,a]+=1
#        else:
#            cfs_mtx[a,p]+=1
#    return cfs_mtx
#
#
#
#
#F=np.array(F)
#for i in range(65536):
#    if F[i]==F.max():
#       print i
#
#2700
#3080
#3200
#3764
#19465
#19625
#20108
#24268  
#
#
#for a in A[3200]:
#    print SFEW_CATE.index(CATE[int( a)])

#    L = np.argmax(label_test,axis=1)
#    P = np.argmax(emotion_model.predict(image_test),axis=1) 
#    sum_info = np.zeros((nb_class,nb_class)) 
#    #col = label row = predict
#    correct=0
#    summ=0
#    #Angry Bored Concentrated Disgust Fear Happy Sad Surprised
#    for i in range(0,len(L)):
#        sum_info[P[i]][L[i]]+=1
#        summ+=1
#        if P[i]==L[i]:
#            correct+=1   
#        else:
#            print(L[i],P[i])      