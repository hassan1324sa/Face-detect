import dlib
import cv2 as c 
import numpy as np
# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cam = c.VideoCapture(0)

while 1: 
    _,frame = cam.read()
    gray =c.cvtColor(frame,c.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        c.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        landmarks = predictor(gray,face)
        for n in range (0,68):
            x,y= landmarks.part(n).x,landmarks.part(n).y
            c.circle(frame,(x,y),3,(0,255,0),-1)
    c.imshow("winname",frame)
    if c.waitKey(60) == 27:
        break
cam.release()
c.destroyAllWindows()