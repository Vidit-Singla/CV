import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam,hcam = 1280,720     #camera width,height

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
volRange = volume.GetVolumeRange()      #-65 to 0+

minVol = volRange[0]
maxVol = volRange[1]

cap = cv2.VideoCapture(0)
cap.set(3,wcam)   #prop 3 is width and 4 is height
cap.set(4,hcam)
ptime = 0

detector = htm.handDetector(detectionCon=0.8)
volBar = 400
vol = 0
volPer = 0

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        x1,y1 = lmList[4][1],lmList[4][2]    #index tip
        x2,y2 = lmList[8][1],lmList[8][2]    #thumb tip

        cx,cy = (x1+x2)//2, (y1+y2)//2   #middle of line

        cv2.circle(img, (x1,y1),10,(200,255,100),cv2.FILLED)
        cv2.circle(img, (x2,y2),10,(200,255,100),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(200,255,100),2)
        cv2.circle(img,(cx,cy),10,(200,255,100),cv2.FILLED)

        length = math.hypot((x2-x1),(y2-y1))   #dist b/w tips

        vol = np.interp(length,[50,260],[minVol,maxVol])    #volume range in terms of distance
        volPer = np.interp(length,[50,260],[0,100])

        volBar = np.interp(length,[50,260],[400,150])
        volume.SetMasterVolumeLevel(vol,None)

        if length < 50:
            cv2.circle(img,(cx,cy),10,(0,255,255),cv2.FILLED)

    cv2.rectangle(img,(50,150), (85,400), (200,255,100), 3)    #volume bar
    cv2.rectangle(img,(50,int(volBar)), (85,400), (0,255,255), 3)
    cv2.putText(img, f'Volume: {int(volPer)}%',(30,500), cv2.FONT_HERSHEY_PLAIN,2,(200,255,100),1)

    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}',(30,70), cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)
    
    
    
    cv2.imshow("Img",img)
    cv2.waitKey(1)