import cv2
import mediapipe as mp
import time
import numpy as np

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, 
                 segmentation=False, detectConf=0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.w, self.h = None, None
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                            model_complexity=self.complexity,
                            smooth_landmarks=self.smooth,
                            enable_segmentation=self.segmentation,
                            min_detection_confidence=self.detectConf,
                            min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        
    #skeleton point 추출(32개의 point) 및 출력
    def findPose(self, frame):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_RGB)
        self.h, self.w, _ = frame.shape
          
        return frame

                
    #skeleton point list 반환(idx, x, y, visibility) 
    def getPoints(self):
        lm = []
        cc = []
        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                #landmark의 결과로 나오는 x,y,z는 각 frame에서의 위치에 대한 비율값이므로 이를 변환
                if idx>29: # cc로 변환하는 경우 
                    break
                cx, cy = int(landmark.x * self.w), int(landmark.y * self.h)
                # 2차원
                lm.append([cx, cy, landmark.visibility])
                        
            cc = self.MP2CC(lm)

        else:
            for _ in range(17):
                cc.append([0, 0, 0.0])

        
        keypoint_arr = np.array(cc,dtype=np.float32)
        keypoint_dict = {'keypoints':keypoint_arr }
                            
        return [keypoint_dict] 
    
    #skeleton world point list 반환(idx, x, y, visibility)
    #엉덩이 쪽 좌표를 0,0,0으로 두고 계산한 2D 공간 좌표
    def getWorldPoints(self):
        lm = []
        cc = []
        if self.results.pose_world_landmarks:
            for idx, landmark in enumerate(self.results.pose_world_landmarks.landmark):
                if idx>29:
                    break

                lm.append([landmark.x, landmark.y, landmark.visibility])

            cc=self.MP2CC(lm)            
            
        else:
            for _ in range(17):
                cc.append([0.0,0.0,0.0])

        coco_array = np.array(cc,dtype=np.float32)
        landmark_dict = {'keypoints':coco_array }
                        
        return [landmark_dict]

    def get3DPoints(self):
        lm = []
        ntu = []
        if self.results.pose_world_landmarks:
            for landmark in self.results.pose_world_landmarks.landmark:
                lm.append([landmark.x, landmark.y, landmark.z])
                        
            ntu = self.MP2NTU(lm)
        else:
            for _ in range(25):
                ntu.append([0.0,0.0,0.0])

        ntu_array = np.array(ntu, dtype=np.float16)
        landmark_dict = {'keypoints':ntu_array }         
        return [landmark_dict]

    def MP2CC(self,lm):
        cc=[]
        idx = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
        for i in idx:
            cc.append(lm[i])
        return cc     

    def MP2NTU(self,lm):
        n = [0 for i in range(26)]

        length = len(lm[0])
        # 몸통&얼굴
        n[1]= [(lm[24][i]+lm[23][i])/2 for i in range(length)]
        n[21] = [(lm[12][i]+lm[11][i])/2 for i in range(length)]
        n[2] = [(n[1][i]+n[21][i])/2 for i in range(length)]
        n[4] = lm[0]
        n[3] = [n[4][i]/3+2*n[21][i]/3 for i in range(length)]
        # 왼팔
        n[5], n[6] , n[7], n[22], n[23] = lm[11], lm[13], lm[15], lm[19], lm[21]
        n[8] = [(n[7][i]+n[22][i])/2 for i in range(length)]
        # 오른팔
        n[9], n[10],n[11], n[24], n[25] = lm[12], lm[14], lm[16], lm[20], lm[22]
        n[12] = [(n[11][i]+n[24][i])/2 for i in range(length)]
        # 왼다리
        n[13],n[14], n[15], n[16]  = lm[23], lm[25], lm[27], lm[31]
        # 오른다리
        n[17], n[18], n[19], n[20] = lm[24], lm[26], lm[28], lm[32]
        
        return n[1:]

#테스트용 함수
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
    video_name = "A3XCZ5Ow6A8_0_45.mp4"
    cap = cv2.VideoCapture("videos/clip video/"+video_name)
    Detector = PoseDetector(complexity=1)  
    

    flag, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fix_w, fix_h = resize_wh(frame)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter("videos/skeleton_added/"+video_name, fourcc, fps, (fix_w, fix_h), True) 
         
    prev_time = 0
    
    while flag:
        resize_frame = cv2.resize(frame, (fix_w, fix_h), interpolation=cv2.INTER_AREA)
        resize_frame = Detector.findPose(resize_frame, False)        
        
        Detector.getWorldPoints(resize_frame, True)
        
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