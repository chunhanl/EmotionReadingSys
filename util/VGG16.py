# -*- coding: utf-8 -*-
# Reference:
#[Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

from keras.layers import Flatten, Dense, Input , MaxPooling2D, Convolution2D,  Activation, Dropout, Concatenate
from keras.models import Model

def create_VGG16( weight_path = None, nb_class = 3):


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
    
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(nb_class, name='fc8')(x)
    x = Activation('softmax', name='fc8/softmax')(x)
    
    #gaussian xavier initial
    
#    flat = Flatten()(pool5)
#    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
#    fc6_drop = Dropout(0.5)(fc6)
#    fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
#    fc7_drop = Dropout(0.5)(fc7)
#    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)
    
    model = Model(img_input, x)
    model.load_weights(weight_path)
    return model


def create_VGG16_nofinal( weight_path = None, nb_class = 3):
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
    
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(nb_class, name='fc8')(x)
    x = Activation('softmax', name='fc8/softmax')(x)
    
    #gaussian xavier initial
    
#    flat = Flatten()(pool5)
#    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
#    fc6_drop = Dropout(0.5)(fc6)
#    fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
#    fc7_drop = Dropout(0.5)(fc7)
#    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)
    
    model = Model(img_input, x)
    model.load_weights(weight_path)
    return model

def create_VGG16_FG( weight_path = None, nb_class = 3):
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
    
    eye_input  = Input(shape=(224, 224, 3))
    nos_input  = Input(shape=(224, 224, 3))
    mou_input  = Input(shape=(224, 224, 3))
    eye  = model(eye_input)
    nose = model(nos_input)
    mouth= model(mou_input)

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
    x = Dense(4096, name='fc7', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu', name='fc7/relu')(x)

    x = Dense(nb_class, name='fc9', kernel_initializer='glorot_normal')(x)
    x = Activation('softmax', name='fc9/softmax')(x)  
    emotion_model = Model([eye_input,nos_input,mou_input], x)       
    emotion_model.load_weights(weight_path)
    return emotion_model





    
def create_VGG16_FG_3Part( weight_path = None, nb_class = 3):
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
    x = Dense(4096, name='fc7', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(4096, name='fc8', kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu', name='fc8/relu')(x)
    x = Dense(nb_class, name='fc9', kernel_initializer='glorot_normal')(x)
    x = Activation('softmax', name='fc9/softmax')(x)         
    emotion_model = Model([eye_input,nos_input,mou_input], x)
    emotion_model.load_weights(weight_path)

    return emotion_model