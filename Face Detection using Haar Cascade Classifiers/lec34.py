import cv2 as c 
import numpy as np 


faceCascade = c.CascadeClassifier('haarcascade_frontalface_default.xml')

cam =c.VideoCapture(0)

while True:
    _,frame =cam.read()
    gray =c.cvtColor(frame,c.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for ( x,y,w,h) in faces:
        c.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
        break 
    c.imshow("img",frame)
    if c.waitKey(1)  == 27  :
        break   
cam.release()
c.destroyAllWindows()