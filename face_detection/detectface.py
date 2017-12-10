import torch
from torch.autograd import Variable
from openfacepytorch.loadOpenFace import prepareOpenFace
from scipy.misc import imresize
import numpy as np

import cv2

import requests, glob

def l2dist(vec1,vec2):
    assert(len(vec1.size())==1)
    assert(len(vec2.size())==1)
    return float(torch.sum((vec1-vec2)**2))


# get faces:
facefiles = glob.glob('faces/*')
facedict = {}
for filename in facefiles:
    facedict.update(torch.load(filename))
print facedict.keys()

# load FaceNet:
model = prepareOpenFace(useCuda=False)
model.eval() # necessary cause of those BatchNorm layers

# initialize some opencv stuff
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

vecs = []
i = 0
hits = {name:0 for name in facedict.keys()}
roi_resized = np.zeros((96,96,3))
while True:
    _, im = cam.read()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(imgray, 1.3, 5)

    for (x,y,w,h) in faces:
        # draws a red rectangle round the face:
        im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        # crop the face
        roi_gray = imgray[y:y+h, x:x+w]
        roi_color = im[y:y+h, x:x+w]
        roi_resized = imresize(roi_color,(96,96))
        # perform histogram equalization:
        roi_ycc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2YCR_CB)
        roi_ycc[:,:,0] = cv2.equalizeHist(roi_ycc[:,:,0])
        roi_resized = cv2.cvtColor(roi_ycc, cv2.COLOR_YCR_CB2BGR)

        # prep image for pytorch:
        face = torch.from_numpy(roi_resized.transpose(2,0,1))
        face = face.unsqueeze(0)
        face = face.type(torch.FloatTensor) / 255.0
        face = Variable(face)

        # get fingerprint
        fingerprint = model(face)[0] # not sure what the second returned value is
        fingerprint = fingerprint.squeeze()

        # find the person whose fingerprint matches this face the closest:
        mindist = 100.0
        for name, facevec in facedict.items():
            dist = l2dist(fingerprint,facevec)
            if dist < mindist:
                bestname = name
                mindist = dist
        hits[bestname] += 1
        
    i += 1
    if i >= 30:
        # send get requests for anyone whose face was detected in more than half of the last 30 frames:
        present_names = [name for name, numhits in hits.items() if numhits > 10]
        print hits
        print "------------------------"
        for name in present_names:
            print "hi %s" % name
            try:
                requests.get("http://192.168.137.68:8080/api/getprofile",params={"personname":name})
            except requests.exceptions.ConnectionError:
                print "connection timed out"
        # reset frame counter and hits counter:
        i = 0
        hits = {name:0 for name in facedict.keys()}
    
    cv2.imshow('im',roi_resized)
    if cv2.waitKey(25) != 255:
        break
cv2.destroyAllWindows()
