# this script crops the area around the lips.

import cv2
from cv2 import resize
from cv2 import imshow
import numpy as np
import os
import sys
import dlib
import pandas as pd
from matplotlib import pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lip_reading/shape_predictor_68_face_landmarks.dat")

def get_rect(shape):
    rw = 0
    rh = 0
    rx = 65535
    ry = 65535
    for (x,y) in shape:
        rw = max(rw,x)
        rh = max(rh,y)
        rx = min(rx,x)
        ry = min(ry,y)
    return (rx,ry,rw-rx,rh-ry)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

inpDir = 'lip_reading/data/'
person = {}
data_type = "words"
img = None
mouth = None
firstImg = None
firstPerson = None
firstSentenceId = None
firstSentenceId2 = None
ratio = 1.0/2.0
pi = 0
linpDir = os.listdir(inpDir)

for personStr in linpDir:
    pi += 1
    print("person: %s [%d/%d]" % (personStr,pi,len(linpDir)))
    person[personStr] = {}
    personFolder = '%s/%s/%s' % (inpDir,personStr, data_type)
    lpersonFolder = os.listdir(personFolder)
    si = 0
    for sentenceId in lpersonFolder:
        si += 1
        person[personStr][sentenceId] = {}
        print("reading sentence %s for person %s [%d/%d,%d/%d]" % (sentenceId,personStr,si,len(lpersonFolder),pi,len(linpDir)))
        sentenceFolder = '%s/%s' % (personFolder,sentenceId)
        for sentenceId2 in os.listdir(sentenceFolder):
            sentenceFolder2 = '%s/%s/%s' % (personFolder,sentenceId,sentenceId2)
            person[personStr][sentenceId][sentenceId2] = {}
            for frame in os.listdir(sentenceFolder2):
                file = "%s/%s" % (sentenceFolder2,frame)
                if(not os.path.isfile(file)):
                    print("%s does not exist" % file)
                    sys.exit(1)
                if(frame[0:5] != "color"):
                    # skip depth data
                    # kinda reminds me on the song "War" of Edwin Starr
                    continue
                frame = frame[6:-4]
                #print("reading file: %s" % file)
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #speed up detector by resizing first
                img2 = cv2.resize(img, None, fx=ratio, fy=ratio)

                
                #detects whole face
                rects = detector(img2, 1)
                if len(rects) == 0:
                    print("error finding head at file: %s" % file)
                    continue
                rects[0] = dlib.scale_rect(rects[0],1/ratio)
                shape = predictor(image=img,box=rects[0])
                shape = shape_to_np(shape)
                
                #indices from 48 to 67 are for mouth
                shape = [shape[x] for x in range(48,68)]
                (x, y, w, h) = get_rect(shape)

                for s in shape:
                    s[0] -= x
                    s[1] -= y
                
                person[personStr][sentenceId][sentenceId2][frame] = {}
                
                mouth = img[y:y+h, x:x+w].copy()
                print("filling: person[%s][%s][%s][%s] with shapes %d" % (personStr,sentenceId,sentenceId2,frame,len(shape)))
                person[personStr][sentenceId][sentenceId2][frame]["mouth"] = mouth
                
                person[personStr][sentenceId][sentenceId2][frame]["shape"] = shape
            

for p in person.keys():
    for sentence in person[p].keys():
        for sentence2 in person[p][sentence].keys():
            os.makedirs("%s/%s/%s/%s" % (data_type, p, sentence, sentence2))
            for id in person[p][sentence][sentence2].keys():
                mouth = person[p][sentence][sentence2][id]["mouth"]
                cv2.imwrite("%s/%s/%s/%s/f%s.png" % (data_type, p, sentence, sentence2, id), mouth)
                

prueba = person["M01"]
person["M01"]["01"]["01"]["001"]["mouth"].shape

for sentences in prueba.keys():
    for id in sentences.keys():
        for frame in id.keys():

plt.imshow(cv2.resize(person["M01"]["01"]["01"]["001"]["mouth"], [28, 10]))
plt.imshow(person["M01"]["01"]["01"]["001"]["mouth"])

plt.show()

