# -*- coding: utf-8 -*-


import cv2
import numpy as np
import math
import tkinter as tk
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
from eye_tracker import *
from monitor_detection import *



##import person_and_phone
import tasklist

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

def findAnyDesk():  
    string1 = 'AnyDesk.exe'

    file1 = open("./taskList.txt", "r")
      

    flag = 0
    index = 0
      

    for line in file1:  
        index += 1 
          
        
        if string1 in line:
            
          flag = 1
          break 
              
    # checking condition for string found or not
    if flag == 0: 
       return 0     ## No  AnyDesk    
    else: 
       return 1     ## Yes AnyDesk
      
    # closing text file    
    file1.close() 

face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX

#Monitor detection
monitor_count = monitor_count()
monitor_flag = 0
#AnyDesk Detection
anyDesk_flag =0
anyDesk_flag = findAnyDesk()

if(monitor_count > 1):
    monitor_flag = 1


#Mouth Detection reqs

outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )


if(monitor_flag == 1) :
    print("Your system consists of multiple monitors")
if(anyDesk_flag == 1):
    print("Presnce of AnyDesk software has been detected")
else:    
        
##########################################################
#Mouth_opening_detector reqsss################################
    while(True):
        ret, mouthImage = cap.read()
        rects = find_faces(mouthImage, face_model)
        for rect in rects:
            shape = detect_marks(mouthImage, landmark_model, rect)
            draw_marks(mouthImage, shape)
            cv2.putText(mouthImage, 'Press r to record Mouth distances', (30, 30), font,
                        1, (0, 255, 255), 2)
            cv2.imshow("Output", mouthImage)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break
    cv2.destroyAllWindows()
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]
    ##########################################################
    while True:
        ret, img = cap.read()
        freshRet, freshImage = cap.read()
        if ret == True:
            faces = find_faces(freshImage, face_model)
            for face in faces:
    ##########################################################
    #eye_tracker################################################
                shape = detect_marks(freshImage, landmark_model, face)
                mask = np.zeros(freshImage.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                
                eyes = cv2.bitwise_and(freshImage, freshImage, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    ##            threshold = cv2.getTrackbarPos('threshold', 'image')
                threshold = 200
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)
                
                eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)

                            
    ##########################################################

    ##########################################################
    #Open_mouth_detector #######################################
                                         
                        
                shape = detect_marks(freshImage, landmark_model, rect)
                cnt_outer = 0
                cnt_inner = 0
                draw_marks(img, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        cnt_outer += 1 
                for i, (p1, p2) in enumerate(inner_points):
                    if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                        cnt_inner += 1
                if cnt_outer > 3 and cnt_inner > 2:
                    print('Mouth open')
                    cv2.putText(img, 'Mouth open', (30, 30), font, 2, (255, 255, 128), 3)




    ##########################################################
    #head_pose_estimation########################################           
                marks = detect_marks(freshImage, landmark_model, face)
##                mark_detector.draw_marks(freshImage, marks, color=(0, 255, 0))
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],      # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                    ], dtype="double")
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                
                
                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose
                
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                
                
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                    
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                    
                    # print('div by zero error')
                if ang1 >= 40:
                    print('Head down')
                    cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                elif ang1 <= -40:
                    print('Head up')
                    cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
                 
                if ang2 >= 38:
                    print('Head right')
                    cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                elif ang2 <= -38:
                    print('Head left')
                    cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                
                cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
    ##########################################################
    ##########################################################
    ##############################################
                           
            cv2.imshow('img', img)
            cv2.imshow('FreshImage', freshImage)

            

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
