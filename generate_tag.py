import cv2
import os
print(cv2.__version__)
print(print(cv2.__file__))

dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict_aruco, parameters)

print('\n'.join(dir(cv2.aruco)))
local_path = os.getcwd()
print(local_path)
num_mark = 10
size_mark = 200  #
dir_mark = os.path.join(local_path, 'tags')
for count in range(num_mark):

    id_mark = count
    img_mark = cv2.aruco.generateImageMarker(dict_aruco, id_mark, size_mark)

    if count < 10:
        img_name_mark = 'mark_id_0' + str(count) + '.jpg'
    else:
        img_name_mark = 'mark_id_' + str(count) + '.jpg'
    path_mark = os.path.join(dir_mark, img_name_mark)

    img = cv2.cvtColor(img_mark, cv2.COLOR_BGR2RGB)

    # cv2.waitKey(0)
    ret = cv2.imwrite(path_mark, img_mark)
    print(dir_mark, ret)