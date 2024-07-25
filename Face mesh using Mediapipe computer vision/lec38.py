import cv2 as c 
import mediapipe as mp

cam = c.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh.FaceMesh()
while True:
    _,frame = cam.read()
    rgb = c.cvtColor(frame , c.COLOR_BGR2RGB)
    result = mpFaceMesh.process(rgb)
    height,width,_ = frame.shape
    for facialLandMarks in result.multi_face_landmarks:
        for i in range(0,468):
            x = int(facialLandMarks.landmark[i].x * width)
            y = int(facialLandMarks.landmark[i].y * height)
            c.circle(frame,(x,y),1,(0,255,0),-1)
    c.imshow("Face Mesh",frame)
    if c.waitKey(1) == 27:
        break
cam.release()
c.destroyAllWindows()