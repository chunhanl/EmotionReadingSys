from tools import calculateScales,detect_face_12net,NMS,filter_face_24net,filter_face_48net
import cv2
import numpy as np
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet

print('Creating Face Detector....')
print('Creating Face Detector Pnet....')
Pnet = create_Kao_Pnet('./weight/MTCNN/12net.h5')
print('Creating Face Detector Rnet....')
Rnet = create_Kao_Rnet('./weight/MTCNN/24net.h5')
print('Creating Face Detector Onet....')
Onet = create_Kao_Onet('./weight/MTCNN/48net.h5')  
print('Face Detector Created')

def detectFace(img, threshold):
    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        ouput = Pnet.predict(input) 
        out.append(ouput)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i][0][0][:, :, 1]  
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = NMS(rectangles, 0.7, 'iou')

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    out = []
    predict_24_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)
        crop_number += 1

    predict_24_batch = np.array(predict_24_batch)
    out = Rnet.predict(predict_24_batch)

    cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
    cls_prob = np.array(cls_prob)  # convert to numpy
    roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
    roi_prob = np.array(roi_prob)
    rectangles = filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    predict_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_batch.append(scale_img)
        crop_number += 1

    predict_batch = np.array(predict_batch)
    output = Onet.predict(predict_batch)
    cls_prob = output[0]
    roi_prob = output[1]
    pts_prob = output[2] 
    rectangles = filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    return rectangles


