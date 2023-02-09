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
FONTSCALE = 0.75
FONTCOLOR = (0, 0, 255)  # BGR
THICKNESS = 1
LINETYPE = 1

@ray.remote
class FileController:
    def __init__(self, args) -> None:
        self.video_path = args.video
        self.label = args.label
        self.fps = args.fps
        self.video_list = [file for file in os.listdir(self.video_path) if file.endswith(".mp4") or file.endswith(".avi")]
        self.short_side = args.short_side
        self.file_cnt = 0
        self.file_names = self.get_file_names()
        self.num_video = len(self.video_list)
        self.out_P_dict = {'S': 1,'P': 0}
        self.state = args.generate_skeleton_file

        
        if self.label <= 2:
            self.flip = True
        else:
            self.flip = False
            
        
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.file_cnt >= self.num_video-1:
            if not self.state or not self.flip:
                raise StopIteration
            else:
                self.flip=False
                self.file_cnt = 0

        frames, file_name = self.read_video(self.file_cnt, self.flip)
        self.file_cnt += 1
        return frames, file_name

    def __len__(self) -> int:
            return self.num_video
    
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
        
        file_name = self.get_out_file_name(frame_name)
        return frames, file_name
    
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
        if self.label <= 2:
            names = [x.split('.')[0] for x in self.video_list]
        else:
            names = [x.split('_')[0] for x in self.video_list]
        return names

    
    
    def get_out_file_name(self, frame_name) -> str:
        if self.label >= 4:
            file_name = frame_name + ".skeleton"
        else:
            if self.out_P_dict['P'] >= 40:
                self.out_P_dict["P"] = 1
                self.out_P_dict["S"] += 1
            else:
                self.out_P_dict['P'] += 1
            file_name = "S{0:03d}C001P{1:03d}R001A{2:03d}.skeleton".format(self.out_P_dict['S'], self.out_P_dict['P'], self.label)
        return file_name
        
    def write_videos(self, idx: int, top1, top3, res_str) -> None:
        video = self.video_list[idx]

        frame_path = 'video_out/' + video
        vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)

        flag, frame = vid.read()
        h, w = frame.shape[:2] 
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter("./videos/out/"+video, fourcc, video_fps, (w, h), True) 
        while True:
            if not flag:
                break

            cv2.putText(frame, top1, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)  
            cv2.putText(frame, res_str, (10, 60), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)   
            # cv2.putText(frame, top3[2], (10, 90), FONTFACE, FONTSCALE,
            #         FONTCOLOR, THICKNESS, LINETYPE)   
            # cv2.putText(frame, top3[2], (10, 90), FONTFACE, FONTSCALE,
            #         FONTCOLOR, THICKNESS, LINETYPE)   
            # cv2.putText(frame, top5[3], (10, 120), FONTFACE, FONTSCALE,
            #         FONTCOLOR, THICKNESS, LINETYPE)   
            # cv2.putText(frame, top5[4], (10, 150), FONTFACE, FONTSCALE,
            #         FONTCOLOR, THICKNESS, LINETYPE)   
            writer.write(frame) 
            flag, frame = vid.read()

        vid.release()
        writer.release()
        
    def get_len(self) -> int:
        return self.num_video

@ray.remote
class SkeletonMaker:
    def __init__(self, args) -> None:
        self.model = pm.remote(complexity=args.complexity,detectConf=0.75, trackConf=0.75)
        
        if args.label == 1: #painting
            self.bodyID = 0
        elif args.label == 2: #interview
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
    
    
    
    
    

    procs = []
        
    # for i in tqdm(range(len(videoFiles))):
    #     proc = Process(target=multiprocess, args=(args, videoFiles, i))
    #     procs.append(proc)
    #     proc.start()
    
    # for proc in procs:
    #     proc.join()
    
    info_list = []
    # pool = Pool(processes=4)
    # for i in tqdm(range(len(videoFiles))):
    #     info_list.append((skeletonMaker, videoFiles, i))
        
    # with tqdm(total=len(videoFiles)) as pbar:
    #     for _ in tqdm(pool.map(multiprocess, info_list)):
    #         pbar.update()
    
    
    
    # pool.map(multiprocess, info_list)
    # pool.close()
    # pool.join()
    videoFiles = FileController.remote(args)
    


    
    # print(cpu_cnt)
    # with Parallel(n_jobs=cpu_cnt, prefer="processes") as parallel:
    #     parallel(delayed(multi_gen)(idx, skeletonMaker=skeletonMaker, videoFiles=videoFiles) for idx in tqdm(range(len(videoFiles))))
    
    skeletonMaker_list = []
    for _ in range(CPU_NUM):
        skeletonMaker = SkeletonMaker.remote(args)
        skeletonMaker_list.append(skeletonMaker)
    
    videoFiles_ray = ray.put(videoFiles)
    
    funcs = []
    
    video_cnt = 0
    video_num = ray.get(videoFiles.get_len.remote())
    if args.label <= 2:
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
    
    # try:  
   
        
        
            # logger.debug("{} 처리 완료".format(videoFiles.video_list[videoFiles.file_cnt-1]))
    # except:
    #     logger.error("{} 처리중 오류 발생".format(videoFiles.video_list[videoFiles.file_cnt])) 
    
@ray.remote     
def write_skeleton_file(skeleton_info: list, file_name: str) -> None:
    frame_dir = './data/mediapipe'
    os.makedirs(frame_dir, exist_ok=True)
    file_dir = frame_dir + "/" + file_name
    
    f = open(file_dir, 'w')
    for idx in range(len(skeleton_info)):
        f.write(skeleton_info[idx])
    f.close()
     
 
@ray.remote  
def multi_gen(idx, skeletonMaker, videoFiles, flip): 
    frames, file_name = ray.get(videoFiles.read_video.remote(idx, flip))
    file_contents, is_valid = ray.get(skeletonMaker.gen_skeleton_file.remote(frames))
    if is_valid:       
        write_skeleton_file.remote(file_contents, file_name)
        print(str(idx)+"th file extract finished: "+file_name)
    # del file_contents, skeletonMaker
    # gc.collect()