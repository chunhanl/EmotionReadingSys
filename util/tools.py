#Ref https://github.com/xiangrufan/keras-mtcnn
#import sys
#from operator import itemgetter
import numpy as np
import cv2
'''
Function:
    change rectangles into squares (matrix version)
Input:
    rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    squares: same as input
'''
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles
'''
Function:
    apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
Input:
    rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'iom':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle
'''
Function:
    Detect face position and calibrate bounding box on 12net feature map(matrix version)
Input:
    cls_prob : softmax feature map for face classify
    roi      : feature map for regression
    out_side : feature map's largest size
    scale    : current input image scale in multi-scales
    width    : image's origin width
    height   : image's origin height
    threshold: 0.6 can have 99% recall rate
'''
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    (x,y) = np.where(cls_prob>=threshold)
    boundingbox = np.array([x,y]).T
    bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1,bb2),axis = 1)
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T
    boundingbox = boundingbox + offset*12.0*scale
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3,'iou')
'''
Function:
    Filter face position and calibrate bounding box on 12net's output
Input:
    cls_prob  : softmax feature map for face classify
    roi_prob  : feature map for regression
    rectangles: 12net's predict
    width     : image's origin width
    height    : image's origin height
    threshold : 0.6 can have 97% recall rate
Output:
    rectangles: possible face positions
'''
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3,'iou')
'''
Function:
    Filter face position and calibrate bounding box on 12net's output
Input:
    cls_prob  : cls_prob[1] is face possibility
    roi       : roi offset
    pts       : 5 landmark
    rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
    width     : image's origin width
    height    : image's origin height
    threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
    rectangles: face positions and landmarks
'''
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    # pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    # pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
    # pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    # pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
    # pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    # pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    # pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
    # pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    # pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
    # pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3,'iom')
'''
Function:
    calculate multi-scale and limit the maxinum side to 1000 
Input: 
    img: original image
Output:
    pr_scale: limit the maxinum side to 1000, < 1.0
    scales  : Multi-scale
'''
def calculateScales(img):
    caffe_img = img.copy()
    pr_scale = 1.0
    h,w,ch = caffe_img.shape
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0/max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    #multi-scale
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

# '''
# Function:
#     calculate switch definition of landmark point to new def
# Input:
#     pts: old definition pts
# Output:
#     pts_new: new def pts
# '''
# def pts_def_rectify(pts):
#     pts_new = np.zeros_like(pts)
#     pts_new[:, 0]= pts[:,0]
#     pts_new[:, 1]= pts[:,5]
#     pts_new[:, 2]= pts[:,1]
#     pts_new[:, 3]= pts[:,6]
#     pts_new[:, 4]= pts[:,2]
#     pts_new[:, 5]= pts[:,7]
#     pts_new[:, 6]= pts[:,3]
#     pts_new[:, 7]= pts[:,8]
#     pts_new[:, 8]= pts[:,4]
#     pts_new[:, 9]= pts[:,9]
#     return pts_new

'''
Function:
    calculate   landmark point , new def
Input:
    cls_prob  : cls_prob[1] is face possibility
    roi       : roi offset
    pts       : 5 landmark
    rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
    width     : image's origin width
    height    : image's origin height
    threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
    rectangles: face positions and landmarks
'''


def filter_face_48net_newdef(cls_prob,roi,pts,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,1]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,3]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,6]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,8]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    # print (pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3,'idsom')
'''
Function:
    calculate mean value of img_list for double checck img quality
Input:
    img_nparray: numpy array of input
Output:
    img_nparray: numpy array of img mean value
'''


def imglist_meanvalue(img_nparray):
    img_mean_array = np.mean(img_nparray ,axis=(1,2,3))
    return np.array(img_mean_array)

'''
Function(C.H.L):
    mark rectangles on the img
Input:
    img: source img, rectangles: mark positions array
Output:
    draw: marked images
'''
def mark_rectangles_score(img, rectangles, score, nb_class=3, dichotomy = False):
    draw = img.copy()

    if nb_class==3:
        for i in range(0,len(rectangles)):
            rectangle = rectangles[i]
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                color = (255*score[i][0], 255*score[i][1], 255*score[i][2])
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), color, 1)
                #text = ['Bored','Happy','Focus']
                #cv2.putText(draw,text[score[i].argmax()], (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_PLAIN, 2, color)
    #            for i in range(5, 15, 2):
    #               cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
    
    elif nb_class==8:
        CATE = [     'Angry'            
                    ,'Bored'
                    ,'Concentra'
                    ,'Disgust'
                    ,'Fear'
                    ,'Happy'
                    ,'Sad'
                    ,'Surprised'
        ]
        if dichotomy:
            CATE_Color = [   [152, 0, 11]  #Red 
                            ,[254, 0, 24]  #Red       
                            ,[11, 76, 0]  #Green
                            ,[152, 0, 11]  #Red 
                            ,[152, 0, 11]  #Red 
                            ,[102,255,102]  #Green
                            ,[152, 0, 11]  #Red 
                            ,[11, 76, 0]  #Green
            ] 
        else:
            CATE_Color = [   [255,0,0]  #Red 
                            ,[0,0,255]  #Blue             
                            ,[0,51,0]#White  
                            ,[191,40,191]#Purple
                            ,[144, 207, 240] #Cyan
                            ,[102,255,102]  #Green
                            ,[0,0,0]    #Black
                            ,[255,255,0]#Yellow
            ]  
         
        for i in range(0,len(rectangles)):
            rectangle = rectangles[i]
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                
                color = (CATE_Color[score[i].argmax()][::-1])                 
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), color, 2)
                cv2.putText(draw,CATE[score[i].argmax()], (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, color)
    return draw

def mark_rectangles_score_AFP(img, rectangles, AFP, score, nb_class=3, dichotomy = False):
    draw = img.copy()

    if nb_class==3:
        for i in range(0,len(rectangles)):
            rectangle = rectangles[i]
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                color = (255*score[i][0], 255*score[i][1], 255*score[i][2])
#                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), color, 1)
                cv2.line(draw, (AFP[i][0][0],AFP[i][1][0]) ,(AFP[i][0][1],AFP[i][1][1]),color,5)
                
    elif nb_class==8:
        CATE = [     'Angry'            
                    ,'Bored'
                    ,'Concentra'
                    ,'Disgust'
                    ,'Fear'
                    ,'Happy'
                    ,'Sad'
                    ,'Surprised'
        ]
        if dichotomy:
            CATE_Color = [   [152, 0, 11]  #Red 
                            ,[254, 0, 24]  #Red       
                            ,[11, 76, 0]  #Green
                            ,[152, 0, 11]  #Red 
                            ,[152, 0, 11]  #Red 
                            ,[102,255,102]  #Green
                            ,[152, 0, 11]  #Red 
                            ,[11, 76, 0]  #Green
            ] 
        else:
            CATE_Color = [   [255,0,0]  #Red 
                            ,[0,0,255]  #Blue             
                            ,[0,51,0]#White  
                            ,[191,40,191]#Purple
                            ,[144, 207, 240] #Cyan
                            ,[102,255,102]  #Green
                            ,[0,0,0]    #Black
                            ,[255,255,0]#Yellow
            ]  

        for i in range(0,len(rectangles)):
            rectangle = rectangles[i]
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                
                color = (CATE_Color[score[i].argmax()][::-1])                 
#                cv2.line(draw, (AFP[i][0][0],AFP[i][1][0]) ,(AFP[i][0][1],AFP[i][1][1]),color,5)
                affinepoints =  np.array(AFP[i], np.int32).reshape((8,1),order='F').reshape((4,2))
                affinepoints =  np.array([affinepoints[0],affinepoints[1],affinepoints[3],affinepoints[2]])
                affinepoints =  affinepoints.reshape((-1,1,2))
                print affinepoints
                cv2.polylines(draw, [affinepoints] , True, color, 2)
                cv2.putText(draw,CATE[score[i].argmax()], (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, color)
    return draw
'''
Function(Frank.C):
    rotate shift to align the face based on the WFLM5 position
Input:
    src: source img, TESTLM5: 5 parts * 2 x,y
Output:
    faces and aligned 5 points
'''
####################read img and show points #######################
def extractAlignedFace(src,TESTLM5):

    imgsize=float(224)
    WFLM5=([[  74.542222,   44.725333],
            [ 161.466667,   43.232   ],
            [ 121.457778,  105.429333],
            [  84.746667,  141.418667],
            [ 157.297778,  140.522667]])
    WFLM54=np.delete(WFLM5,2,0)
    WFLM54t=WFLM54
    WFLM54t=WFLM54t.astype(np.float32)
    TESTLM54=np.delete(TESTLM5,2,0)
#    TESTLM5x=TESTLM54[:,0]
#    TESTLM5y=TESTLM54[:,1]            
#    plt.figure("test")    
#    plt.subplot(1,2,1)
#    plt.imshow(img[:,:,::-1])
#    plt.scatter(TESTLM5x,TESTLM5y,c='red',s=1,label='red',alpha=0.6,edgecolors='white')
    
    ################## get transformation ###############################
    TESTLM54t=TESTLM54
    TESTLM54t=TESTLM54t.astype(np.float32)
    H=cv2.getPerspectiveTransform(TESTLM54t,WFLM54t)
    Hbb=cv2.getPerspectiveTransform(WFLM54t,TESTLM54t)#matrix of boundingbox
    Wbb=np.array([[0,0],[0,224],[224,0],[224,224]])
    Wbb=Wbb.astype(np.float32)
    afpoint=cv2.perspectiveTransform(TESTLM5[None,:,:],H)
    afpoint=afpoint[0]
    
    afbb=cv2.perspectiveTransform(Wbb[None,:,:],Hbb)
    afbb=afbb[0]
    afbbx=afbb[:,0]
    afbby=afbb[:,1]
    ################## plot               ###################  
    dstimg = cv2.warpPerspective(src,H,(int(imgsize),int(imgsize)))
    return dstimg[:,:,::-1], afpoint, [afbbx,afbby]
#    plt.subplot(1,2,2)
#    plt.imshow(dstimg[:,:,::-1])
#    plt.scatter(afpointx,afpointy,c='red',s=1,label='red',alpha=0.6,edgecolors='white')
#    plt.show()
    

'''
Function(C.H.L):
    crop faces out of the img
Input:
    img: source img, rectangles: crop positions array,rectangles[i][0:3] is the position
Output:
    faces: face images arrays
'''
def extract_faces(img, rectangles, align =False):
    faces = []
    align_reces =[]
    afine_points=[]
    for rec in rectangles:
        if align:
            rec = np.asarray(rec)
            crop_img, face_rec, afpoint = extractAlignedFace(img[:,:,::-1],rec[5:15].reshape(5,2)) 
            faces.append( crop_img )
            align_reces.append(face_rec)
            afine_points.append(afpoint)

        else:
            y1=int(rec[0])
            x1=int(rec[1])
            y2=int(rec[2])
            x2=int(rec[3])
            crop_img = img[ x1:x2,y1:y2,:]
            faces.append( crop_img )
    if align:
        return faces,align_reces,afine_points
    else:
        return faces