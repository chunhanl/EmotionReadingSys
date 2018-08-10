# -*- coding: utf-8 -*-
# C.H.Lu 2017.09, Academic Sinica
#a3232012a@gmail.com 

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from keras.preprocessing import image
import threading
import tensorflow as tf
import time
from util.tools import extract_faces,mark_rectangles_score_AFP
from util.detector import detectFace
from util.VGG16 import create_VGG16_FG_3Part

CATE    = ['Angry' ,'Bored','Concentrated','Disgust'   ,'Fear'  ,'Happy'      ,'Sad'   ,'Surprised']
C_POINTS= [  -0.25 , -1    , +0.5         , -0.25      , -0.25  ,   +1        ,  -0.25 ,   +0.5    ]
C_COLOR = [  'r'   ,'b'    , '#003300'    , '#BF28BF'  ,  'c'   , '#66FF66'   , 'k'    ,   'y'     ]
  
def subprocess():
    global REC,AFP,SCORE    
    global IMG
    global isRECupdate
    global isSUBrunning
    global graph            
    global nb_class
    global isLegend
    global num_Face
    global plot_score
    global CATE,C_COLOR, C_POINTS
    
    with graph.as_default():
        TIME_start= time.time()
        REC = detectFace(IMG,threshold) 
        face_rec=[]
        faces, face_rec, AFP = extract_faces(IMG, REC, align=True) #faces = BGR
        finegrain_faces=[]
        
        for i in range(0, len(faces)):
            # Obtain MTCNN face detection matrix face_rec[i]
            tmp_rec = face_rec[i]
            eye  = max(tmp_rec[0,1] ,tmp_rec[1,1] )
            nos  = tmp_rec[2,1]
            mou  = max(tmp_rec[3,1],tmp_rec[4,1])
            # Decide spliting y axis
            spliy1= int( eye + ( nos - eye ) /2 )
            spliy2= int( nos + ( mou - nos ) /2 )
            # Obtain extracted face
            X = faces[i]
            X = image.img_to_array(X)    
            # Zero-center by mean pixel (BGR)
            X[:, :, 0] -= 103.939
            X[:, :, 1] -= 116.779
            X[:, :, 2] -= 123.68
            X = np.expand_dims(X, axis=0)
            # Standardize
            std = np.std(X, axis=(0, 1, 2))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[2] = X.shape[3]
            std = np.reshape(std, broadcast_shape)
            X /= (std + 1e-8)  
            # Split into 3 parts
            X1 = X[0][:][0:spliy1][:]
            X1 = cv2.resize(X1,(224,224))
            X2 = X[0][:][spliy1:spliy2][:]
            X2 = cv2.resize(X2,(224,224))
            X3 = X[0][:][spliy2::][:]
            X3 = cv2.resize(X3,(224,224))  
            # Add into finegrain_faces for further emo detection
            finegrain_faces.append([X1,X2,X3])
        finegrain_faces = np.asarray(finegrain_faces)
        # Predict Emotions
        SCORE =[]
        SCORE.append(VGG16_emo.predict([finegrain_faces[:,0,...],finegrain_faces[:,1,...],finegrain_faces[:,2,...]]))
        # Calculating FPS
        TIME_end= time.time()
        fps = 1.0/(TIME_end-TIME_start)
        # Calculating Emotions and Score
        if len(SCORE)!=0:
            S2 = argmax(np.asarray(SCORE)[0,:,:],axis=1)
            count = np.zeros(nb_class)
            for i in range(0,len(count)):
                count[i] = np.count_nonzero(S2==i)
                if i>=nb_class: print('Unknown predict :' + str(i))
            num_Face.append( count[:] )
            plot_score.append( np.sum(count * C_POINTS)/np.sum(count)  )                      
              
        # Plotting graph
        x1_1 = np.asarray( range(0,len(num_Face)) )
        tmp = np.resize( num_Face, (len(num_Face), nb_class))
        tmp2= plot_score
        if( len(tmp2)==len(x1_1) ):
            ax1.clear()
            ax1.set_ylim([-1,1])
            ax1.plot(x1_1, np.zeros(len(x1_1)), c='k', ls='--', lw=2)
            ax1.plot(x1_1, np.asarray(tmp2), c='c', ls='-', lw=3, label='EmotionScore')                   
            ax1.legend(loc=3)  
            ax1.text(0, 1, r'Faces: ${}$ detectFPS:${}$'.format(len(REC),round(fps,2)), fontsize=12)
    
            num_POS = tmp[:,2]*C_POINTS[2] + tmp[:,5]*C_POINTS[5]  +tmp[:,7]*C_POINTS[7] 
            num_NEG = tmp[:,0]*C_POINTS[0] + tmp[:,1]*C_POINTS[1]  +tmp[:,3]*C_POINTS[3] +tmp[:,4]*C_POINTS[4] + tmp[:,6]*C_POINTS[6]             
            ax2.clear()
            ax2.plot(x1_1, num_POS, c='g', ls='-', lw=3, label='Score of POSITIVE') 
            ax2.plot(x1_1, num_NEG*-1, c='r', ls='-', lw=3, label='Score of NEGATIVE')   
            ax2.legend(loc=3) 
            plt.pause(0.001)
            plt.draw()     
                        
        isRECupdate = True
        isSUBrunning=False      
       
        

if __name__ == '__main__' :
       
    REC=[]
    IMG=[]
    AFP=[]
    SCORE = []
    rectangles=[]
    affinepoints=[]
    score = []    
    isRECupdate=False
    isSUBrunning=False
    nb_class=8
    threshold = [0.5,0.5,0.6] 
    # Tensor @ Different Thread requires 'with graph.as_default():'
    graph = tf.get_default_graph() 
    # Create part-based VGG16
    VGG16_emo = create_VGG16_FG_3Part('./weight/weights_partbasedVGG16.hdf5',nb_class = 8)
    # Read test video
    cap = cv2.VideoCapture(0)
    if not cap.open('./data/video/v1.avi'):
        print 'Failed to read input video'
    else:     
        
        fig = plt.figure('Info')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        isLegend= False
    
        num_Face =[]
        plot_score=[]
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        
        
        while(True):
            A = time.time()
            ret, img = cap.read()
    
            if ret == False:
                #Replay Video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)       
            else:
                img= cv2.resize(img,(960,540))
                IMG=img #BGR
                if not isSUBrunning:
                    t= threading.Thread(target=subprocess)
                    t.start()
                    isSUBrunning = True
                    
                if isRECupdate: 
                    rectangles  = REC
                    affinepoints= AFP
                    score = SCORE
                    isRECupdate=False                          
                    
                S = np.array(score)
                if len(S)!=0 :
                    S =  S[0,:,:]
                    
                draw = mark_rectangles_score_AFP(img, rectangles, affinepoints,S , nb_class = nb_class, dichotomy = True)
                cv2.imshow('Emo_System', draw)
    
                key = cv2.waitKey(50)
                if  key & 0xFF == ord('q'):
                    break;
                elif  key  & 0xFF == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    num_Face =[]
                    plot_score=[]
                    fig.clear()
                    fig = plt.figure('Info')
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)
                    isLegend= False
                
    #Release Memory
    cap.release()
    cv2.destroyAllWindows()
    import gc
    VGG16_emo = None
    Pnet = None
    Onet = None
    Rnet = None
    for i in range(5): gc.collect()

