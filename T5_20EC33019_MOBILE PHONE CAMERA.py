# Camera Caliberation performed wth mobile phone camera
# Contributed By Kaushal Jadhav 
# Roll No.-20EC30019
# First Year Undergraduate,Dept. of Electronics and Electrical Communication Engineering, IIT Kharagpur

# Geometric camera calibration, also referred to as camera resectioning, estimates the parameters of a lens and image sensor of an image or video camera.
# This is then used to calculate lens distortion or the position of camera in 3-D world. 
# Camera Caliberation is often used in the application of stereo vision where the camera projection matrices of two cameras are used to 
# calculate the 3D world coordinates of a point viewed by both cameras.

# Import necessary libraries
import numpy as np 
# Numpy module for handling matrices efficiently

import cv2

from cv2 import aruco
# Aruco library is used for Aruco Markers. These are special markers which provide enough correspondences for camera pose estimation.

cor_list=[]
ids_list=[]
count=[]
chk=True

# The aruco module includes some predefined dictionaries covering a range of different dictionary sizes and marker sizes
# Here importing aruco dictionary of standard 6X6 size and creating the board
aruco_dict=aruco.Dictionary_get(aruco.DICT_6X6_250)

'''
An ArUco Board is a set of markers that acts like a single marker in the sense that it provides a single pose for the camera.
The difference between a Board and a set of independent markers is that the relative position between the markers in the Board is known a priori. 
This allows that the corners of all the markers can be used for estimating the pose of the camera respect to the whole Board.
The main benefits of using Boards are:
1. The pose estimation is much more versatile. Only some markers are necessary to perform pose estimation. 
   Thus, the pose can be calculated even in the presence of occlusions or partial views.
2. The obtained pose is usually more accurate since a higher amount of point correspondences (marker corners) are employed.
'''
board=aruco.GridBoard_create(3,4,0.018,0.004,aruco_dict)

# Get images for caliberation
# 25 different images had been obtained through mobile phone camera for caliberation
for c in range(25):
 img=cv2.imread('mob\img'+str(c+1)+'.jpg')

 # Conversion to grayscale image is necessary.
 img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 img.astype(np.uint8)

 #Detect aruco_markers from the given image. 
 #Corners is a list of detected marker-corners
 # With the help of corners and their IDs the intrinsic and extrinsic parameters of the camera can be obtained
 corners,ids,rejected_pts=aruco.detectMarkers(img,aruco_dict)

 # Refine the detection
 corners,ids,rejected_pts,recovered_ids=aruco.refineDetectedMarkers(img,board,corners,ids,rejected_pts)

 # Append the corners and ids into a stack
 if(chk):        # Initialise stack
     cor_list=corners
     ids_list=ids
     chk=False
 else:
     cor_list=np.vstack((cor_list,corners))
     ids_list=np.vstack((ids_list,ids))
 count.append(len(ids))          # Get the number of ids for each image    

h=img.shape[0]
w=img.shape[1]

count=np.array(count)          # Convert to numpy array which is required for calibrateCameraAruco function

#Calibrate the camera
ret,cam_mat,dist_coeff,rvecs,tvecs=aruco.calibrateCameraAruco(cor_list,ids_list,count,board,(h,w),None,None)

# Some info about param
# @ cam_mat is the intrinsic camera matrix
# @ dist_coeff is for distorted images.
# @ rvecs is the rotation vector
# @ tvecs is the translation vector

print(cam_mat)
print(rvecs)
print(tvecs)

'''
Code for documentation 
'''
w=open('a a aruco.txt','w')
w.write('mobile phone camera:\n')
w.write('corners:')
w.write(str(cor_list))
w.write('\n')
w.write('ids_list:')
w.write(str(ids_list))
w.write('\n')
w.write('Intrinsic Camera Matrix')
w.write('\n')
w.write(str(cam_mat))
w.write('\n')
w.write('Extrinsic Camera Parameters')
w.write('\n')
w.write('R_vecs')
w.write('\n')
w.write(str(rvecs))
w.write('\n')
w.write('T_vecs')
w.write('\n')
w.write(str(tvecs))
w.write('\n')
w.write('Shape of the image=')
w.write(str(img.shape[0])+','+str(img.shape[1]))
w.close()

print(img.shape[0],img.shape[1])