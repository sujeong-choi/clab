import PoseModule as pm
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Convert RGB video to skeleton pkl file')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('--fps', type=int, default=10, help='frame extraction count per sec')
    parser.add_argument('--short-side', type=int, default=480, help='specify the short-side length of the image')
    parser.add_argument('--complexity', type=int, default=1, choices=range(0, 3), help='Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.')
    parser.add_argument('--label', type=int, default=121)
    args = parser.parse_args()
    return args

class FileController:
    def __init__(self, args) -> None:
        self.video_path = args.video
        self.fps = args.fps
        self.video_list = [file for file in os.listdir(self.video_path) if file.endswith(".mp4")]
        self.num_video = len(self.video_list)
        self.short_side = args.short_side
        self.file_cnt = 0
        self.file_names = self.get_file_names()
        
        self.prev_frame_name = ""
        self.out_file = ""
        self.out_P_cnt = 0
        self.label = args.label
        self.out_P_dict = dict()
        
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.file_cnt >= self.num_video:
            raise StopIteration
        frames, frame_name = self.read_video(self.file_cnt)
        
        if frame_name not in self.out_P_dict or self.out_P_dict[frame_name]['S'] >= 32:
            self.out_P_dict[frame_name] = dict(S=18,P=self.out_P_cnt)
            self.out_P_cnt += 1
        else:
            self.out_P_dict[frame_name]['S'] += 1
        

        
        self.file_cnt += 1
        return frames, frame_name

    def __len__(self) -> int:
        return self.num_video
    
    def read_video(self, idx: int) -> tuple([list, str]):
        video = self.video_list[idx]
        frame_name = self.file_names[idx]

        vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        skip_frame = float(video_fps / self.fps)
        skip_cnt = 0
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
                skip_cnt = 0

            if new_h is None:
                new_w, new_h = self.resize_wh(frame, self.short_side)
            
            frame =  cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
        frame_dir = './out'
        os.makedirs(frame_dir, exist_ok=True)
        file_dir = frame_dir + "/" + self.get_out_file_name(frame_name)
        
        f = open(file_dir, 'w')
        for idx in range(len(skeleton_info)):
            f.write(skeleton_info[idx])
        f.close()
        
        
    
    def get_out_file_name(self, frame_name) -> str:
        file_name = "S{0:03d}C001P{1:03d}R001A{2:03d}.skeleton".format(self.out_P_dict[frame_name]['S'], self.out_P_dict[frame_name]['P'], self.label)
        return file_name


class SkeletonMaker:
    def __init__(self, args) -> None:
        self.model = pm.PoseDetector(complexity=args.complexity)
        self.bodyID = 100000000000000000
        
    def gen_skeleton_file(self, frames) -> list:
        file = []
        
        skeleton_list, score_list = self.skeleton_inference(frames)
        frame_num = len(skeleton_list)
        file.append(self.translate_str(frame_num))
        self.bodyID += 1
        
        for idx, skeleton in enumerate(skeleton_list):
            body_cnt = 1
            file.append(self.translate_str(body_cnt))
            
            
            body_info = self.get_body_info(score_list[idx])
            file.append(self.translate_str(body_info))
            
            joint_num = 25
            file.append(self.translate_str(joint_num))
            
            for i in range(joint_num):
                if len(score_list[idx]) == 0 or score_list[idx][i] <= 0.6:
                    other_joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]  #차후 수정 가능
                else:
                    other_joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2]
                file.append(self.translate_str(skeleton[i] + other_joint))
        
        return file
                    
                
        
    def skeleton_inference(self, frames: list) -> tuple([list, list]):
        ret = []
        score_list = []
        for frame in frames:
            frame = self.model.findPose(frame)
            point_dict, score = self.model.get3DPoints()
            ret.append(point_dict)
            score_list.append(score)
        return ret, score_list
    

    def get_body_info(self, score_list: list) -> list:
        body_info = []
        
        body_info.append(self.bodyID)
        
        #해당 함수 이후 차후 수정 가능
        #clipedEdges
        body_info.append(0)
        
        #handLeftConfidence
        if len(score_list) == 0 or score_list[7] <= 0.6:
            body_info.append(0)
            body_info.append(0)
        else:
            body_info.append(1)
            body_info.append(1)
        
        #handRightConfidence
        if len(score_list) == 0 or score_list[1] <= 0.6:
            body_info.append(0)
            body_info.append(0)
        else:
            body_info.append(1)
            body_info.append(1)
        
        #isResticted
        body_info.append(0)
        
        #lean_info
        body_info.append(0)
        body_info.append(0)
        
        #trackingState
        score_avg = np.mean(score_list)
        if score_avg > 0.3:
            body_info.append(2)
        else:
            body_info.append(0)
        
        
        return body_info 
        
        
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
            
    
def main():
    args = parse_args()
    videoFiles = FileController(args)
    skeletonMaker = SkeletonMaker(args)
    
    for frames, frame_name in tqdm(videoFiles):
        file_contents = skeletonMaker.gen_skeleton_file(frames)        
        videoFiles.write_skeleton_file(file_contents, frame_name)
        
if __name__ == "__main__":
    main()