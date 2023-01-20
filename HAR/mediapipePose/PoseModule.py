import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, 
                 segmentation=False, detectConf=0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.detectConf = detectConf
        self.trackConf = trackConf
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                            model_complexity=self.complexity,
                            smooth_landmarks=self.smooth,
                            enable_segmentation=self.segmentation,
                            min_detection_confidence=self.detectConf,
                            min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        
    #skeleton point 추출(32개의 point) 및 출력
    def findPose(self, frame, draw=True):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        self.results = self.pose.process(frame_RGB)
        
        if  self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame,  self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return frame

                
    #skeleton point list 반환(idx, x, y, z, visibility)
    def getPoints(self, frame, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                        #landmark의 결과로 나오는 x,y,z는 각 frame에서의 위치에 대한 비율값이므로 이를 변환
                        h, w, c = frame.shape
                        cx, cy = int(landmark.x*w), int(landmark.y*h)
                        landmark_list.append([idx, cx, cy, int(landmark.z), float(landmark.visibility)])
                        
                        if draw:
                            cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)
                            
        return landmark_list
    
    #skeleton world point list 반환(idx, x, y, z, visibility)
    #엉덩이 쪽 좌표를 0,0,0으로 두고 계산한 3D 공간 좌표
    def getWorldPoints(self, frame, draw=True):
        landmark_list = []
        idx_to_coordinates = {}
        if self.results.pose_world_landmarks:
            for idx, landmark in enumerate(self.results.pose_world_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int((landmark.x*w/2)+w/2), int((landmark.y*h/2)+h/2)
                landmark_list.append([idx, cx, cy, landmark.visibility])
                if landmark.visibility < self.detectConf:
                    continue
                
                landmark_px = [cx, cy]
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px
            if draw:      
                num_landmarks = len(self.results.pose_world_landmarks.landmark)
                for connection in self.mpPose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                        raise ValueError(f'Landmark index is out of range. Invalid connection '
                                        f'from landmark #{start_idx} to landmark #{end_idx}.')
                    if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                        cv2.line(frame, idx_to_coordinates[start_idx],
                                idx_to_coordinates[end_idx], (224, 224, 224), 2)
                    
                        
        return landmark_list

def resize_wh(frame, short_side=256):
    h, w = frame.shape[:2]  
    
    if w >= h:
        fix_w = int(short_side*float(w/h))
        fix_h = short_side
    else:
        fix_w = short_side
        fix_h = int(short_side*float(h/w))


    return fix_w, fix_h
    

def main():
    video_name = "sample_pass1.mp4"
    cap = cv2.VideoCapture("videos/"+video_name)
    Detector = PoseDetector(complexity=0)  
    

    flag, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fix_w, fix_h = resize_wh(frame)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter("outs/"+video_name, fourcc, fps, (fix_w, fix_h), True) 
         
    prev_time = 0
    
    while flag:
        resize_frame = cv2.resize(frame, (fix_w, fix_h), interpolation=cv2.INTER_AREA)
        resize_frame = Detector.findPose(resize_frame)        
        
        # point_list = Detector.getPoints(frame)
        # Detector.getWorldPoints(frame)
        
        cur_time = time.time()
        fps = 1/(cur_time-prev_time)
        prev_time = cur_time

        cv2.putText(resize_frame, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN, 4 ,(255,0,0),3)

        cv2.imshow("Frame", resize_frame)
        writer.write(cv2.resize(resize_frame, (fix_w, fix_h)))
        cv2.waitKey(1)
        flag, frame = cap.read()
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()