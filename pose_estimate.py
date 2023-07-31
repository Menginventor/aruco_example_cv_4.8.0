import cv2
import os
import yaml
import numpy as np
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

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash



dir_config = os.path.join(local_path, 'config\\camera_calibration')
with open(dir_config) as file:
    cam_calib_dist = yaml.load(file, Loader=yaml.Loader)
    print('load yaml')
    print(cam_calib_dist.keys())
    mtx = cam_calib_dist['mtx']
    dist = cam_calib_dist['dist']

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]


    cv2.imshow('raw',frame)
    #cv2.imshow('undistort', dst)


    arucoParams = detector.getDetectorParameters
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(dst)
    if not markerIds is None:
        for idx in range(len(markerIds)):
            rvecs, tvecs, trash = my_estimatePoseSingleMarkers(markerCorners[idx], 5.3, newcameramtx, dist)

            print(tvecs[0][2])
    cv2.aruco.drawDetectedMarkers(dst, markerCorners, markerIds)
    # print(markerIds)
    cv2.imshow('detect', dst)




    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
