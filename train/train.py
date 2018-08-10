# -*- coding: utf-8 -*-
# C.H.Lu 2017.09, Academic Sinica
# a3232012a@gmail.com 
# Reference:
# [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
# courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder


import os
import sys
import cv2
import numpy as np
import glob
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten, Dense, Input , MaxPooling2D, Convolution2D, Activation, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from tqdm import tqdm
from util.tools import extractAlignedFace
from util.detector import detectFaceExhautiveSearch
sys.path.append('../')
pretrained_vgg ='./weight/vggface_tf_notop.h5'
weight_savepath='./weight/'
SFEW_dataroot = './data/SFEW/'
Crawler_dataroot = './data/Crawler/' 
Rafd_dataroot = './data/RaFD/'    
CATE =      ['Angry','Neutral','Disgust'   ,'Fear'  ,'Happy'  ,'Sad','Surprise']
SFEW_CATE=  ['Angry','Disgust','Fear'      ,'Happy' ,'Neutral','Sad','Surprise']

nb_class = 7  
def finegrain_generator(in_images, in_labels, batchsize = 8, state='Train'):         
    if state=='Train':
        datagen = image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        channel_shift_range=50,
        horizontal_flip=True)    
    elif state=='Val':
        datagen = image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        channel_shift_range=0,
        horizontal_flip=False)              
    else:
        print 'Unrecognizable state'
        return
    

    datagen.fit(in_images)
    while True:
        imgs,labels = datagen.flow(in_images,in_labels,batch_size=batchsize,shuffle=True).next()        
        imgFG = []
        lblFG = []
        for img,label in zip(imgs,labels):
            y1 = label[1]
            y2 = label[2]
            y3 = label[3]
            X1 = img[:][0:y2][:]
            X2 = img[:][y1:y3][:]
            X3 = img[:][y2::][:]
            X1 = cv2.resize(X1,(224,224))
            X2 = cv2.resize(X2,(224,224))
            X3 = cv2.resize(X3,(224,224))
            imgFG.append([X1,X2,X3])

        imgFG=np.array(imgFG)
        lblFG=labels[:,0]
        lblFG = np_utils.to_categorical(lblFG,nb_class)
        yield [imgFG[:,0,:,:,:],imgFG[:,1,:,:,:],imgFG[:,2,:,:,:]],lblFG


#LOAD DATA FROM ALL################################################################################################
def getTrainData():  
    f  = open(os.path.join(Crawler_dataroot,'datasetM.txt'),'r')
    f1 = open(os.path.join(SFEW_dataroot,'Train','datasetM.txt'),'r')
    f2 = open(os.path.join(SFEW_dataroot,'Val'  ,'datasetM.txt'),'r')
    f3 = open(os.path.join(Rafd_dataroot,'datasetM.txt'),'r')
    list_crawl      = f.readlines()   
    list_sfewTrain  = f1.readlines()   
    list_sfewVal    = f2.readlines()  
    list_rafd       = f3.readlines()  
    list_all =[]
    list_all.append({'src':'Crawler'   , 'list':list_crawl})
    list_all.append({'src':'SFEW_Train', 'list':list_sfewTrain})
    list_all.append({'src':'SFEW_Val'  , 'list':list_sfewVal})
    list_all.append({'src':'Rafd'      , 'list':list_rafd})
    
    image_train=[]
    label_train=[]  
    class_sum = np.zeros(nb_class)
    
    for datas in list_all:
        
        if datas['src']=='Crawler' :
            root = os.path.join(Crawler_dataroot ,'SOURCE')
        elif datas['src']=='SFEW_Train':
            root = os.path.join(SFEW_dataroot ,'Train')
        elif datas['src']=='SFEW_Val':
            root = os.path.join(SFEW_dataroot ,'Val')
        elif datas['src']=='Rafd':
            root = os.path.join(Rafd_dataroot )
        
        for imginfo in tqdm(datas['list']):     
            label  = imginfo.split(' ')[1]
            if label=='Bored' or label=='Concentrated':
                label='Neutral'
            if label=='Surprised' or label=='Concentrated':
                label='Surprise'
            label_idx   = CATE.index(label)   
            rec    = np.array(imginfo[imginfo.find('[')+1:imginfo.find(']')].split(','),dtype=float)
            srcimg = imginfo.split(' ')[2] 
            imgpath= (datas['src']=='Crawler' and os.path.join(root,srcimg)) or os.path.join(root,label,srcimg)

            X  = image.load_img( imgpath ) 
            X  = image.img_to_array(X)
            #Align     
            X, face_rec = extractAlignedFace(X[:,:,::-1],rec[5:15].reshape(5,2)) 
            X = np.expand_dims(X, axis=0) 
            X = preprocess_input(X)
            X = np.array(X)        
            eyey1= face_rec[0,1] 
            eyey2= face_rec[1,1] 
            nosy = face_rec[2,1]
            mouy1= face_rec[3,1]
            mouy2= face_rec[4,1]
                      

            image_train.append( X[0,:,:,:] )
            label_train.append( [label_idx,int(np.mean((eyey1,eyey2))),int(nosy),int(np.mean((mouy1,mouy2)))] )  
            class_sum[ label_idx ]+=1

    image_train = np.array(image_train)
    label_train = np.array(label_train)

    print 'ClassSum:{}'.format(class_sum) 
    for c in class_sum:
        if c==0:
            print'ClassSum cant be zero'
            break
    class_sum /= (class_sum.min())
    class_weight = 1/class_sum    
    print 'ClassWeight:{}'.format(class_weight)
    return image_train,label_train, class_weight
    


def getTestData():
    
    source_images_test   = glob.glob(os.path.join(SFEW_dataroot,'Test', '*.*'))
    source_images_test.sort()
    ans = open(os.path.join(SFEW_dataroot,'assume_ans'))
    ans = ans.readlines()    
    images,labels=[],[]
    for imgpath in source_images_test:
        X  = image.load_img( imgpath ) 
        images.append( X )
        for info in ans:
            if imgpath.find(info.split('\t')[0]) !=-1:
                sfew_label = SFEW_CATE[int(info.split('\t')[1].split('\n')[0])]
                labels.append( sfew_label  )
                break
           
    fg_images=[]
    fg_labels=[]
    for img,lbl in tqdm(zip(images,labels)):
        X  = image.img_to_array(img)
        X,face_rec,facefind = detectFaceExhautiveSearch(X)
            
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
            fg_images.append([X1,X2,X3])
        else:
            X = np.zeros((224,224,3))
            X = np.array(X)      
            fg_images.append([X,X,X])
        fg_labels.append( CATE.index(lbl) )  
    fg_images = np.array(fg_images)
    fg_labels = np_utils.to_categorical(fg_labels,nb_class)
    return fg_images,fg_labels




    
def getFGModel(fc_secondlast,fc_last):
    alltrain=True
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

    model0.load_weights(pretrained_vgg)
    model1.load_weights(pretrained_vgg)
    model2.load_weights(pretrained_vgg)
    model0.trainable = alltrain
    model1.trainable = alltrain
    model2.trainable = alltrain
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
    emotion_model.summary()    
    return emotion_model


def train(name, fc_secondlast, fc_last, epoch, save_period):
#DATASET################################################################################################          
    print 'Preparing Train Images...'
    image_train,label_train, class_weight = getTrainData() 
    print 'Preparing Test Images...'
    image_test ,label_test                = getTestData()

#MODEL################################################################################################              
    emotion_model = getFGModel( fc_secondlast, fc_last )

#OPTIMIZER################################################################################################          
    SGDoptim = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#    RMSprop = RMSprop(lr=0, rho=0.9, epsilon=1e-08, decay=0.0)
#    Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    emotion_model.compile(loss='categorical_crossentropy', optimizer=SGDoptim, metrics=['accuracy']) 
 
#LOAD PARAM################################################################################################   
#    emotion_model.load_weights('./TRAIN11_7/weights.1599-1.43.hdf5')    

#MODEL SAVER################################################################################################    
    assert os.path.isdir(weight_savepath), 'Weight Save Directory does not exist'
    if not os.path.isdir(os.path.join(weight_savepath,str(name))) :
        os.mkdir(os.path.join(weight_savepath,str(name)))
    checkpointer = ModelCheckpoint(filepath=os.path.join(weight_savepath+'/'+str(name)+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5') ,verbose=1,save_best_only=False, save_weights_only=True, period=save_period) 

#TRAIN#######################################################################################################   

    train_gen = finegrain_generator(image_train, label_train, batchsize = 16, state='Train')
    history_callback = emotion_model.fit_generator(train_gen, steps_per_epoch=len(image_train)/16, epochs=epoch, verbose=1, callbacks=[checkpointer], validation_data=([image_test[:,0,:,:,:],image_test[:,1,:,:,:],image_test[:,2,:,:,:] ],label_test), class_weight=class_weight, initial_epoch=0)
    loss_history = history_callback.history["loss"]
    numpy_loss_history = np.array(loss_history)
    np.savetxt(os.path.join(weight_savepath,str(name),"_loss_history.txt"), numpy_loss_history, delimiter=",")
    
#EVALUATE################################################################################################      
#    emotion_model.load_weights('')
#    ans = emotion_model.predict(images)
#    emotion_model.evaluate(image_val,label_val)    

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', type=str , required=True,help='enter the save dir name')
    parser.add_argument('--fc_secondlast', type=int, required=True, help='enter the second last layer')
    parser.add_argument('--fc_last', type=int, required=True, help='enter the last layer')
    parser.add_argument('--epoch', type=int, required=True)    
    parser.add_argument('--save_period', type=int, required=True)   
    args = parser.parse_args()

    print args
    train(name=args.dir, fc_secondlast=args.fc_secondlast, fc_last=args.fc_last, epoch=args.epoch, save_period=args.save_period)
