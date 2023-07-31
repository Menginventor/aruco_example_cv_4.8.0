import cv2
import os
print(cv2.__version__)
print(print(cv2.__file__))
from matplotlib import pyplot as plt

local_path = os.getcwd()

dir_mark = os.path.join(local_path, 'boards')

path_mark = os.path.join(dir_mark, 'charuco_board.jpg')

dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_aruco, parameters)



squareLength = 40   # Here, our measurement unit is centimetre.
markerLength = 30   # Here, our measurement unit is centimetre.
board = cv2.aruco.CharucoBoard((5,7), squareLength,markerLength,dict_aruco)
board_img = board.generateImage((500, 700))
ret = cv2.imwrite(path_mark, board_img)
