import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('videos/1.mp4')
prev_time = 0

while True:
    #mediapipe은 컬러 이미지를 처리
    flag, frame = cap.read()
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #skeleton point 추출(32개의 point)
    results = pose.process(frame_RGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            #landmark의 결과로 나오는 x,y,z는 각 frame에서의 위치에 대한 비율값이므로 이를 변환
            h, w, c = frame.shape
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)
            
            
            
    # print(results.pose_landmarks)
    
    cur_time = time.time()
    fps = 1/(cur_time-prev_time)
    prev_time = cur_time

    cv2.putText(frame, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,0,0),3)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
