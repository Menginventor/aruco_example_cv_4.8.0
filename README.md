# aruco_example
Instruction
1. Modify parameter and run "create_charuco_board.py"
2. Make charuco board by printing charuco_board.py on A4 pater
3. Run "charuco_cam_capture.py" and press 'c' to capture image of aruco board in difference angle
4. Run "charuco_cam_calib" to calculate camera and distortion mareix ans store as yaml file in '/config'
5. Run "generate_tag.py" and print tag from folder '/tags'
6. Run "pose_estimate.py" to see result. (note. tag size need to be match with real tag size, unit of tag size will be the same as pose value)
