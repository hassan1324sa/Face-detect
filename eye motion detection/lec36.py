import cv2 as c 
import numpy as np

cam =c.VideoCapture(0)

while True:
    _,frame =cam.read()
 #   rigon of interst the important place in video
        #y1,y2  ,X1  ,x2
    roi = frame[269:795,537:1416]
    cols , rows,_ = roi.shape
    gray = c.cvtColor(roi,c.COLOR_BGR2GRAY)
    gray = c.GaussianBlur(gray ,(7,7),0)
    _,thresh = c.threshold(roi,3,255,c.THRESH_BINARY_INV)
    contours,_ =c.findContours(thresh,c.RETR_TREE,c.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours ,key=lambda x: c.contourArea(x) , reverse=True)    
    for contour in contours :
        (x,y,w,h) =c.boundingRect(contour)
        c.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
        c.line(roi,(x+int(w/2),0),(x+int(w/2),rows),(0,255,0),2)
        c.line(roi,(y+int(h/2),0),(cols,y+int(h/2)),(0,255,0),2)
        break
    c.imshow("gray",gray)
    if c.waitKey(60) == 27 :
        break

cam.release()
c.destroyAllWindows()