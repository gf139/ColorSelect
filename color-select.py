import cv2
import numpy as np

def empty(b):
    pass
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


cap=cv2.VideoCapture(0)
cv2.namedWindow("a")
cv2.resizeWindow("a",700,250)
cv2.createTrackbar("Hue min","a",0,179,empty)
cv2.createTrackbar("Hue max","a",179,179,empty)
cv2.createTrackbar("Sat min","a",0,255,empty)
cv2.createTrackbar("Sat max","a",255,255,empty)
cv2.createTrackbar("Val min","a",0,255,empty)
cv2.createTrackbar("Val max","a",255,255,empty) #一样的是为了

while True:
    success,img=cap.read()
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min=cv2.getTrackbarPos("Hue min","a")
    h_max=cv2.getTrackbarPos("Hue max","a")
    s_min=cv2.getTrackbarPos("Sat min","a")
    s_max=cv2.getTrackbarPos("Sat max","a")
    v_min=cv2.getTrackbarPos("Val min","a")
    v_max=cv2.getTrackbarPos("Val max","a")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)    #面具调节所要图形范围
    imgresult=cv2.bitwise_and(img,img,mask=mask)


    imgstack=stackImages(0.6,([img,mask,imgresult]))
    cv2.imshow("imgstack",imgstack)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


