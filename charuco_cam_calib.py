import cv2
import os
import numpy as np
import yaml
local_path = os.getcwd()

dir_mark = os.path.join(local_path, 'boards')

dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_aruco, parameters)



squareLength = 40   # Here, our measurement unit is centimetre.
markerLength = 30   # Here, our measurement unit is centimetre.
board = cv2.aruco.CharucoBoard((5,7), squareLength,markerLength,dict_aruco)
#print('\n'.join(dir(board)))
print(board.getChessboardCorners())
charuco_objp = np.array(board.getChessboardCorners())
print(charuco_objp.shape)
images = np.array([os.path.join(dir_mark, f )for f in os.listdir(dir_mark) if f.endswith(".jpg") and f.startswith('img') ])
print(images)


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    objp = np.zeros((4 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    board_detector = cv2.aruco.CharucoDetector(board)


    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        char_corners, char_ids, markerCorners, markerIds = board_detector.detectBoard(frame)


        if len(markerCorners)>0:
            # SUB PIXEL DETECTION

            objpoints.append(charuco_objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, char_corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (6, 4), corners2, True)
            cv2.imshow('img', frame)
            cv2.waitKey(1000)

            imgpoints.append(corners2)




    imsize = gray.shape[::-1]
    print(objpoints[0])
    print(imgpoints[0])
    return objpoints, imgpoints,imsize

objpoints, imgpoints,imsize = read_chessboards(images)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,imsize, None, None)
print(ret)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
camera_calibration_dict = {
    'ret':ret,
    'mtx':mtx,
    'dist':dist,
    'rvecs':rvecs,
    'tvecs':tvecs
}

dir_config = os.path.join(local_path, 'config\\camera_calibration')
with open(dir_config, 'w') as file:
    documents = yaml.dump(camera_calibration_dict, file)