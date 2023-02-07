from .PoseModule import PoseDetector as pm
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import logging

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (0, 0, 255)  # BGR
THICKNESS = 1
LINETYPE = 1

def parse_args():
    parser = argparse.ArgumentParser(description='Convert RGB video to skeleton pkl file')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('--fps', type=int, default=10, help='frame extraction count per sec')
    parser.add_argument('--short-side', type=int, default=480, help='specify the short-side length of the image')
    parser.add_argument('--complexity', type=int, default=1, choices=range(0, 3), help='Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.')
    parser.add_argument('--label', type=int, default=0)
    args = parser.parse_args()
    return args

class FileController:
    def __init__(self, args) -> None:
        self.video_path = args.video
        self.fps = args.fps
        self.video_list = [file for file in os.listdir(self.video_path) if file.endswith(".mp4") or file.endswith(".avi")]
        self.short_side = args.short_side
        self.file_cnt = 0
        self.file_names = self.get_file_names()
        self.num_video = len(self.video_list)
        self.prev_frame_name = ""
        self.out_file = ""
        self.out_P_cnt = 1
        self.label = args.label
        self.out_P_dict = dict()
        
        if self.label == 2:
            self.flip = False
        else:
            self.flip = True
            
        
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.file_cnt >= self.num_video:
            if not self.flip:
                raise StopIteration
            else:
                self.flip=False
                self.file_cnt = 0

        frames, frame_name = self.read_video(self.file_cnt, self.flip)
        self.file_cnt += 1
        return frames, frame_name

    def __len__(self) -> int:
        if self.label == 2:
            return self.num_video
        else:
            return self.num_video*2
    
    def read_video(self, idx: int, is_flip: bool=False) -> tuple([list, str]):
        video = self.video_list[idx]
        frame_name = self.file_names[idx]

        vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        if self.fps>=video_fps:
            self.fps = video_fps
        skip_frame = float(video_fps / self.fps)
        skip_cnt = 1
        frame_cnt = 0
        frames = []
        # frame_paths = []
        new_h, new_w = None, None

        while True:
            flag, frame = vid.read()
            if not flag:
                break
            
            if skip_cnt < skip_frame:
                skip_cnt += 1
                continue
            else:
                skip_cnt = 1

            if new_h is None:
                new_w, new_h = self.resize_wh(frame, self.short_side)
                
            
            frame =  cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if is_flip:
                cv2.flip(frame, 1)
            frames.append(frame)
            
            # frame_path = frame_tmp.format(frame_cnt + 1)
            # frame_paths.append(frame_path)

            # cv2.imwrite(frame_path, frame)
            frame_cnt += 1
            
        return frames, frame_name
    
    def resize_wh(self, frame, short_side: int) -> tuple([int, int]):
        h, w = frame.shape[:2]  
        
        if w >= h:
            fix_w = int(short_side*float(w/h))
            fix_h = short_side
        else:
            fix_w = short_side
            fix_h = int(short_side*float(h/w))


        return fix_w, fix_h


    def get_file_names(self) -> list:
        names = [x.split('_')[0] for x in self.video_list]
        return names

    
    def write_skeleton_file(self, skeleton_info: list, frame_name: str) -> None:
        frame_dir = './data/mediapipe'
        os.makedirs(frame_dir, exist_ok=True)
        file_dir = frame_dir + "/" + self.get_out_file_name(frame_name)
        
        f = open(file_dir, 'w')
        for idx in range(len(skeleton_info)):
            f.write(skeleton_info[idx])
        f.close()
        
        
    
    def get_out_file_name(self, frame_name) -> str:
        if frame_name not in self.out_P_dict or self.out_P_dict[frame_name]['S'] >= 32:
            self.out_P_dict[frame_name] = dict(S=1,P=self.out_P_cnt)
            self.out_P_cnt += 1
        else:
            self.out_P_dict[frame_name]['S'] += 1
        file_name = "S{0:03d}C001P{1:03d}R001A{2:03d}.skeleton".format(self.out_P_dict[frame_name]['S'], self.out_P_dict[frame_name]['P'], self.label)
        return file_name

    def write_videos(self, idx: int, top1, top5) -> None:
        video = self.video_list[idx]

        frame_path = 'video_out/' + video
        vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)

        flag, frame = vid.read()
        h, w = frame.shape[:2] 
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter("videos/out/"+video, fourcc, video_fps, (w, h), True) 
        while True:
            if not flag:
                break

            cv2.putText(frame, top1, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)  
            cv2.putText(frame, top5[1], (10, 60), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)   
            cv2.putText(frame, top5[2], (10, 90), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)   
            cv2.putText(frame, top5[3], (10, 120), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)   
            cv2.putText(frame, top5[4], (10, 150), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)   
            writer.write(frame) 
            flag, frame = vid.read()

        vid.release()
        writer.release()


class SkeletonMaker:
    def __init__(self, args) -> None:
        self.model = pm(complexity=args.complexity,detectConf=0.75, trackConf=0.75)
        
        if args.label == 0: #painting
            self.bodyID = 0
        elif args.label == 1: #interview
            self.bodyID = 50000
        else:
            self.bodyID = 100000 #pause
        self.non_person = []
        for _ in range(25):
                self.non_person.append([0.0,0.0,0.0])
        
        
    def gen_skeleton_file(self, frames) -> tuple([list, bool]):
            file = []
            
            skeleton_list, score_list, avg_score_list = self.skeleton_inference(frames)
            if sum(avg_score_list)/len(avg_score_list) < 0.75:
                return [], False
            
            frame_num = len(skeleton_list)
            file.append(self.translate_str(frame_num))
            self.bodyID += 1           
            
            for idx, skeleton in enumerate(skeleton_list):
                # print("skeleton = ",skeleton)
                if skeleton!=self.non_person: # sekeleton data가 있을때,
                    body_cnt = 1
                    file.append(self.translate_str(body_cnt))
                    
                    file.append(self.translate_str(self.bodyID))
                    
                    joint_num = 25
                    file.append(self.translate_str(joint_num))
                    
                    for i in range(joint_num):
                        file.append(self.translate_str(skeleton[i]))
                else: # skeleton data가 모두 0으로 채워져있을때
                    file.append('0\n') 
            
            return file, True                    
                
        
    def skeleton_inference(self, frames: list) -> tuple([list, list]):
        ret = []
        score_list = []
        avg_score_list = []
        for frame in frames:
            self.model.findPose(frame)
            landmarks, visibility_list, avg_visibility = self.model.getUpperWorldPoints()
            ret.append(landmarks)
            score_list.append(visibility_list)
            avg_score_list.append(avg_visibility)
        return ret, score_list, avg_score_list     
        
        
    def translate_str(self, data):
        if isinstance(data, int):
            return str(data) + '\n'
        elif isinstance(data, list):
            ret = ""
            for idx in range(len(data)):
                tmpStr = str(data[idx])
                ret += tmpStr + " "
            return ret + "\n"
        else:
            raise

def make_skeleton_data_files(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
    
    
    file_handler = logging.FileHandler('./output.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    
    
    
    videoFiles = FileController(args)
    skeletonMaker = SkeletonMaker(args)
    try:  
        for frames, frame_name in tqdm(videoFiles):
            file_contents, is_valid = skeletonMaker.gen_skeleton_file(frames)
            if is_valid:       
                videoFiles.write_skeleton_file(file_contents, frame_name)
                logger.debug("{} 처리 완료".format(videoFiles.video_list[videoFiles.file_cnt-1]))
    except:
        logger.error("{} 처리중 오류 발생".format(videoFiles.video_list[videoFiles.file_cnt])) 
    