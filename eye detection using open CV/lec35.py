import cv2 as c 
import numpy as np 


faceCascade = c.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade =c.CascadeClassifier('haarcascade_eye.xml') # when u back
cam =c.VideoCapture(0)

while True:
    _,frame =cam.read()
    gray =c.cvtColor(frame,c.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    eyes = eyeCascade.detectMultiScale(gray,2.3,4)
    for ( x,y,w,h) in faces:
        c.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
        c.putText(frame,"Face",(x,y-4),c.FONT_HERSHEY_COMPLEX,1,(255,255,255))
        for ( x,y,w,h) in eyes:
            roi = gray[x:w,y:h]
           
            c.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
            c.putText(frame,"eye",(x,y-4),c.FONT_HERSHEY_COMPLEX,1,(255,255,255))
    c.imshow("img",frame)
    if c.waitKey(1)  == 27  :
        break
cam.release()
c.destroyAllWindows()