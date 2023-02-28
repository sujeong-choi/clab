import cv2
import mediapipe as mp
import time
import numpy as np
import os
from tqdm import tqdm
import ray
@ray.remote
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

    #skeleton point list 반환(키포인트 리스트, 신뢰도 리스트, 신뢰도 평균) 
    #is_draw가 ture일시 입력받은 frame에서 keypoint 출력
    def getPoints(self, frame, is_draw=False):
        lm = []
        visibility_list = []
        if self.results.pose_landmarks:
            if is_draw:
                self.mpDraw.draw_landmarks(frame,  self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                #landmark의 결과로 나오는 x,y,z는 각 frame에서의 위치에 대한 비율값이므로 이를 변환
                cx, cy = int(landmark.x * self.w), int(landmark.y * self.h)
                # 2차원
                lm.append([cx, cy])
                visibility_list.append(landmark.visibility)
        else:
            for _ in range(33):
                lm.append([0, 0])
                visibility_list.append(0.0)
                       
        return lm, visibility_list, sum(visibility_list)/len(visibility_list)
    
    #skeleton world point list 반환(키포인트 리스트, 신뢰도 리스트, 신뢰도 평균)
    #엉덩이 쪽 좌표를 0,0,0으로 두고 계산한 3D 공간 좌표
    def getWorldPoints(self):
        lm = []
        visibility_list = []
        if self.results.pose_world_landmarks:
            for idx, landmark in enumerate(self.results.pose_world_landmarks.landmark):
                lm.append([landmark.x, landmark.y, landmark.z])
                visibility_list.append(landmark.visibility)
            
        else:
            for _ in range(33):
                lm.append([0.0,0.0,0.0])
                visibility_list.append(0.0)

                        
        return lm, visibility_list, sum(visibility_list)/len(visibility_list)
    
    #skeleton world point list 반환(키포인트 리스트, 신뢰도 리스트, 신뢰도 평균)
    #엉덩이 쪽 좌표를 0,0,0으로 두고 계산한 상체의 3D 공간 좌표
    #skeleton world point list return (keypoint list, reliability list, reliability average)
    #3D spatial coordinates of the upper body calculated with the hip coordinates as 0,0,0
    def getUpperWorldPoints(self):
        lm = []
        visibility_list = []
        if self.results.pose_world_landmarks:
            for idx, landmark in enumerate(self.results.pose_world_landmarks.landmark):
                if idx >= 25:
                    break
                if landmark.visibility >= 0.75:
                    lm.append([landmark.x, landmark.y, landmark.z])
                else:
                    lm.append([0.0,0.0,0.0])
                visibility_list.append(landmark.visibility)
            
        else:
            for _ in range(25):
                lm.append([0.0,0.0,0.0])
                visibility_list.append(0.0)

                        
        return lm, visibility_list, sum(visibility_list)/len(visibility_list)
    
    def get3DPoints(self):
        lm = []
        ntu = []
        landmark_score = []
        if self.results.pose_world_landmarks:
            for landmark in self.results.pose_world_landmarks.landmark:
                lm.append([landmark.x, landmark.y, landmark.z])
                landmark_score.append(landmark.visibility)
                        
            ntu = self.MP2NTU(lm)
            landmark_score = self.MP2NTUScore(landmark_score)
        else:
            for _ in range(25):
                ntu.append([0.0,0.0,0.0])
        return ntu, landmark_score
    
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
        n[2] = [(lm[1][i]+lm[21][i])/2 for i in range(length)]
        n[4] = [(lm[2][i]+lm[5][i])/2 for i in range(length)]
        n[3] = [lm[4][i]/3+2*lm[21][i]/3 for i in range(length)]
        # 왼팔
        n[5], n[6] , n[7], n[22], n[23] = lm[11], lm[13], lm[15], lm[19], lm[21]
        n[8] = [(lm[7][i]+lm[22][i])/2 for i in range(length)]
        # 오른팔
        n[9], n[10],n[11], n[24], n[25] = lm[12], lm[14], lm[16], lm[20], lm[22]
        n[12] = [(lm[11][i]+lm[24][i])/2 for i in range(length)]
        # 왼다리
        n[13],n[14], n[15], n[16]  = lm[23], lm[25], lm[27], lm[31]
        # 오른다리
        n[17], n[18], n[19], n[20] = lm[24], lm[26], lm[28], lm[32]
        
        return n[1:]
    
    def MP2NTUScore(self,score_list):
        n = [0 for i in range(26)]
        
        # 몸통&얼굴
        n[1]= (score_list[24]+score_list[23])/2
        n[21]= (score_list[12]+score_list[11])/2       
        n[2]= (score_list[1]+score_list[21])/2       
        n[4]= (score_list[2]+score_list[5])/2        
        n[3]= (score_list[4]+score_list[21]*2)/3
        # 왼팔
        n[5], n[6] , n[7], n[22], n[23] = score_list[11], score_list[13], score_list[15], score_list[19], score_list[21]
        n[8]= (score_list[7]+score_list[22])/2
        # 오른팔
        n[9], n[10],n[11], n[24], n[25] = score_list[12], score_list[14], score_list[16], score_list[20], score_list[22]
        n[12]= (score_list[11]+score_list[24])/2
        # 왼다리
        n[13],n[14], n[15], n[16]  = score_list[23], score_list[25], score_list[27], score_list[31]
        # 오른다리
        n[17], n[18], n[19], n[20] = score_list[24], score_list[26], score_list[28], score_list[32]
        
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

def make_skeleton_videos(args):
    video_path = args.video
    video_list = [file for file in os.listdir(video_path) if file.endswith(".mp4") or file.endswith(".avi")]
    out_path = video_path + '/skelton_added_videos/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    Detector = PoseDetector(complexity=1, detectConf=0.5, trackConf=0.5)
    # Detector = PoseDetector(complexity=1)
    
    for idx in tqdm(range(len(video_list))):
        try:
            visibility_list = []
            cap = cv2.VideoCapture(video_path + '/' + video_list[idx])

            flag, frame = cap.read()
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fix_w, fix_h = resize_wh(frame)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            writer = cv2.VideoWriter(out_path + video_list[idx], fourcc, fps, (fix_w, fix_h), True) 
                
            prev_time = 0
            
            while flag:
                resize_frame = cv2.resize(frame, (fix_w, fix_h), interpolation=cv2.INTER_AREA)
                Detector.findPose(resize_frame)        
                Detector.getPoints(resize_frame, is_draw=True)
                lm, _, avg_visbility = Detector.getUpperWorldPoints()

                cur_time = time.time()
                fps = 1/(cur_time-prev_time)
                prev_time = cur_time
                cv2.putText(resize_frame, str(int(fps)),(30,50),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,0,0),3)

                visibility_list.append(avg_visbility)
                cv2.putText(resize_frame, str(float(avg_visbility)),(30,130),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,0,0),3)  

                writer.write(cv2.resize(resize_frame, (fix_w, fix_h)))
                flag, frame = cap.read()
        except:
            print(video_path + '/' + video_list[idx] +' <- 오류가 있는 파일입니다')
            continue
        
        print(video_list[idx]+":"+str(sum(visibility_list) / len(visibility_list)))
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
     