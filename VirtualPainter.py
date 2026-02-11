import numpy as np
import cv2
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)    #access header imgs
overlayList = []    #stores all images

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]    #default red
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

drawcolor = (0, 0, 255)
brushThickness = 15
eraserThickness = 75
xp,yp = 0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)   

detector = htm.handDetector(detectionCon=0.85)


while True:
    #1. Import image
    success, img = cap.read()
    img = cv2.flip(img,1)   #flip horizontally

    #2. Find Hand Landmark
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:

        #tip of index finger
        x1,y1 = lmList[8][1:]
        #tip of middle finger
        x2,y2 = lmList[12][1:]

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        


        #4. If selection mode  
        if fingers[1] == 1 and fingers[2] == 1:
            xp,yp = 0,0
            # print("Selection Mode")

            if y1<125:     #inside header
                if 250<=x1<=450:
                    header = overlayList[1]
                    drawcolor = (0, 0, 255)
                    
                elif 550<=x1<=750:
                    header = overlayList[2]
                    drawcolor = (255,0,0)
                    
                elif 800<=x1<=950:
                    header = overlayList[3]
                    drawcolor = (0, 255, 0)
                    
                elif 1050<=x1<=1200:
                    header = overlayList[4]
                    drawcolor = (0, 0, 0)
                    
                # elif 130<=x1<=200:
                #     header = overlayList[0]
                #     drawEnabled = False
                    

            
            cv2.rectangle(img,(x1-35,y1-30),(x2+35,y2+30), drawcolor, cv2.FILLED)
          
                
                
        #5. If drawing mode
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1,y1),15,drawcolor,cv2.FILLED)
            # print("Drawing Mode")
            if(xp == 0 and yp == 0):    #starting point
                xp,yp = x1,y1
            if drawcolor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawcolor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawcolor, eraserThickness)
            else:   
                cv2.line(img, (xp,yp), (x1,y1), drawcolor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawcolor, brushThickness)

            xp,yp = x1,y1       #update strating point

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)   #create a mask of white on black
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #setting header
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)    #blend both images
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)

