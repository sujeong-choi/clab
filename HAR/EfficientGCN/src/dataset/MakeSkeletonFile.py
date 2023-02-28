from .PoseModule import PoseDetector as pm
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
import logging
import multiprocessing as mp
from joblib import Parallel, delayed
import ray

CPU_NUM = 30

ray.init(num_cpus=CPU_NUM, dashboard_port=8888)
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (0, 0, 255)  # BGR
THICKNESS = 1
LINETYPE = 1



class FileController:
    def __init__(self, args) -> None:
        if args.video.endswith(".mp4") or args.video.endswith(".avi"):
            self.video_path = args.video[:-4]
            self.isFolder = False
            self.video_list = [args.video]
            self.vid = cv2.VideoCapture(args.video)
        else:
            self.video_path = args.video
            self.isFolder = True
            self.video_list = [file for file in os.listdir(self.video_path) if file.endswith(".mp4") or file.endswith(".avi")]
            
        self.label = args.label
        self.fps = args.fps
        self.short_side = args.short_side
        self.file_cnt = 0
        self.file_names = self.get_file_names()
        self.num_video = len(self.video_list)
        self.out_P_dict = {'S': 0,'P': 1}
        self.state = args.generate_skeleton_file
        self.writer = None

        
        if self.label >= 121:
            self.flip = True
        else:
            self.flip = False
            
        
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.isFolder:
            if self.file_cnt >= self.num_video-1:
                if not self.state or not self.flip:
                    raise StopIteration
                else:
                    self.flip=False
                    self.file_cnt = 0

            frames, file_name, _ = self.read_video(self.file_cnt, self.flip)
            self.file_cnt += 1
            return frames, file_name
        else:
            frames, file_name, isEnd = self.read_video(0)
            if isEnd:
                raise StopIteration
            
            return frames, file_name

    def __len__(self) -> int:
            return self.num_video
    
    def read_video(self, idx: int, is_flip: bool=False) -> tuple([list, str]):
        if self.isFolder:
            video = self.video_list[idx]
            frame_name = self.file_names[idx]

            self.vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = self.vid.get(cv2.CAP_PROP_FPS)
        if self.fps>=video_fps:
            self.fps = video_fps
        # print(str(self.fps)+" fps입니다.")
        skip_frame = float(video_fps / self.fps)
        # print(str(skip_frame)+" skip입니다.")
        
        skip_cnt = 1
        frame_cnt = 0
        frames = []
        # frame_paths = []
        new_h, new_w = None, None
        self.isEnd = False
        while True:
            flag, frame = self.vid.read()
            if not flag:
                self.isEnd = True
                break
            
            if self.fps * 4 <= frame_cnt:
                # print(str(frame_cnt)+"프레임 카운트")
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
        if self.isFolder:
            file_name = self.get_out_file_name(frame_name)
        else:
            file_name = ""
        return frames, file_name, self.isEnd
    
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
        if self.label >= 121: # 2/11
            names = [x.split('.')[0] for x in self.video_list]
        else:
            names = [x.split('_')[0] for x in self.video_list]
        return names

    
    
    def get_out_file_name(self, frame_name) -> str:
        if self.label <= 120: # 2/11
            file_name = frame_name + ".skeleton"
        else:
            if self.out_P_dict['S'] >= 32:
                self.out_P_dict["S"] = 1
                self.out_P_dict["P"] += 1
            else:
                self.out_P_dict['S'] += 1
            file_name = "S{0:03d}C001P{1:03d}R001A{2:03d}.skeleton".format(self.out_P_dict['S'], self.out_P_dict['P'], self.label)
        return file_name
        
    def write_videos(self, idx: int, top1, res_str, frames) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if self.isFolder:
            video = self.video_list[idx]
            out_foler = './videos/t1o_2/'
            vid = cv2.VideoCapture(self.video_path + '/' +video)
            video_fps = vid.get(cv2.CAP_PROP_FPS)

            flag, frame = vid.read()
            h, w = frame.shape[:2] 
            
            if not os.path.exists(out_foler) : os.mkdir(out_foler)
            writer = cv2.VideoWriter(out_foler+video, fourcc, video_fps, (w, h), True) 
            while True:
                if not flag:
                    break

                cv2.putText(frame, top1, (10, 40), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)  
                cv2.putText(frame, res_str, (10, 80), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)   
                writer.write(frame) 
                flag, frame = vid.read()

            vid.release()
            writer.release()
        else:
            h, w = frames[0].shape[:2]
            if top1 == "painting" : FONTCOLOR = (0,0,255)
            else : FONTCOLOR = (0,255,0)
            if self.writer is None: 
                video_fps = self.vid.get(cv2.CAP_PROP_FPS)
                out_foler = './videos/'
                self.writer = cv2.VideoWriter(out_foler+"testing.mp4", fourcc, 30, (w, h), True) 
                print(self.fps)
            for frame in frames:
                if top1 != "others":
                    cv2.putText(frame, res_str, (int(w*0.7), int(h*0.1)), FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)  
                # cv2.putText(frame, res_str, (10, 80), FONTFACE, 1,
                #         FONTCOLOR, THICKNESS, LINETYPE)   
                self.writer.write(frame) 
                
            if self.isEnd:
                self.vid.release()
                self.writer.realse()
        
    def get_len(self) -> int:
        return self.num_video

class SkeletonMaker:
    def __init__(self, args) -> None:
        self.model = pm.remote(complexity=args.complexity,detectConf=0.75, trackConf=0.75)
        
        if args.label == 121: #painting
            self.bodyID = 0
        elif args.label == 122: #interview
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
            self.model.findPose.remote(frame)
            landmarks, visibility_list, avg_visibility = ray.get(self.model.getUpperWorldPoints.remote())
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
    
    videoFiles = FileController.remote(args)
    
    skeletonMaker_list = []
    for _ in range(CPU_NUM):
        skeletonMaker = SkeletonMaker.remote(args)
        skeletonMaker_list.append(skeletonMaker)
    
    videoFiles_ray = ray.put(videoFiles)
    
    funcs = []
    
    video_cnt = 0
    video_num = ray.get(videoFiles.get_len.remote())
    if args.label > 120:
        flip = True
    else:
        flip = False
        
    while True:
        if video_cnt >= video_num:
            if flip:
                flip = False
                video_cnt = 0
            else:
                break
        
        for idx in range(CPU_NUM):
            if video_cnt+idx >= video_num:
                break
            funcs.append(multi_gen.remote(video_cnt+idx, skeletonMaker_list[idx], videoFiles, flip))
        video_cnt += CPU_NUM
            
    ray.get(funcs)
    ray.shutdown()
    

def write_skeleton_file(skeleton_info: list, file_name: str) -> None:
    frame_dir = './data/mediapipe'
    os.makedirs(frame_dir, exist_ok=True)
    file_dir = frame_dir + "/" + file_name
    
    f = open(file_dir, 'w')
    for idx in range(len(skeleton_info)):
        f.write(skeleton_info[idx])
    f.close()
     
 

def multi_gen(idx, skeletonMaker, videoFiles, flip): 
    frames, file_name = ray.get(videoFiles.read_video.remote(idx, flip))
    file_contents, is_valid = ray.get(skeletonMaker.gen_skeleton_file.remote(frames))
    if is_valid:       
        write_skeleton_file.remote(file_contents, file_name)
        print(str(idx)+"th file extract finished: "+file_name)
    # del file_contents, skeletonMaker
    # gc.collect()