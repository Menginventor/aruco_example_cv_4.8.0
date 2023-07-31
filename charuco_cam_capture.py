import cv2
import os
print(cv2.__version__)
print(print(cv2.__file__))
from matplotlib import pyplot as plt

local_path = os.getcwd()

dir_mark = os.path.join(local_path, 'boards')



dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_aruco, parameters)



squareLength = 40   # Here, our measurement unit is centimetre.
markerLength = 30   # Here, our measurement unit is centimetre.
board = cv2.aruco.CharucoBoard((5,7), squareLength,markerLength,dict_aruco)
camera = cv2.VideoCapture(0)
img_idx = 0
while True:
    ret, frame = camera.read()
    cv2.imshow('cam',frame)
    board_detector = cv2.aruco.CharucoDetector(board)



    if cv2.waitKey(2) & 0xFF == ord('c'):
        print('captured')
        path_mark = os.path.join(dir_mark, 'img_%d.jpg' % img_idx)
        img_idx +=1
        ret = cv2.imwrite(path_mark, frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    arucoParams = board_detector.getDetectorParameters
    char_corners, char_ids, markerCorners, markerIds = board_detector.detectBoard(frame)
    #print(ret)
    cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    #print(markerIds)
    cv2.imshow('plot', frame)