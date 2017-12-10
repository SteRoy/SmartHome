import torch
from torch.autograd import Variable
from openfacepytorch.loadOpenFace import prepareOpenFace
from scipy.misc import imresize

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
for i in range(1000):
    _, im = cam.read()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(imgray, 1.3, 5)

    if len(faces) == 0:
        continue
    x,y,w,h = faces[0]

    # crop face:
    im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
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

    vecs.append(fingerprint)

    cv2.imshow('im',roi_resized)
    print cv2.waitKey(25)
cv2.destroyAllWindows()


vecs = torch.stack(vecs)
meanvec = torch.mean(vecs,dim=0)
# output vectors are normalized to lie on the surface of a hypersphere, so best to renormalize meanvec so that it does too:
meanvec /= torch.sqrt(torch.sum(meanvec**2))