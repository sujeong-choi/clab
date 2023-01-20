import PoseModule as pm
import multiprocessing as mp
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import mmcv
from mmcv import dump
from tqdm import tqdm
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Convert RGB video to skeleton pkl file')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('--dimension', type=bool, default=True, help='skeleton dimension: True(default)->3D, False->2D')
    parser.add_argument('--fps', type=int, default=10, help='frame extraction count per sec')
    parser.add_argument('--short-side', type=int, default=480, help='specify the short-side length of the image')
    parser.add_argument('--complexity', type=int, default=1, choices=range(0, 3), help='Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.')
    parser.add_argument('--num_process', type=int, default=1, help='process core number for execute')
    parser.add_argument('--test_scale', type=float, default=0.25, help='test, train ratio. default=0.25, range= 0.0 ~ 1.0')
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
        self.label = self.get_label()
        

    def read_video(self, idx: int) -> dict:
        video = self.video_list[idx]
        frame_name = osp.basename(osp.splitext(video)[0])
        frame_dir = osp.join('./tmp', frame_name)
        os.makedirs(frame_dir, exist_ok=True)
        frame_tmp = osp.join(frame_dir, 'img_{:06d}.jpg')

        vid = cv2.VideoCapture(self.video_path + '/' +video)
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        skip_frame = float(video_fps / self.fps)
        skip_cnt = 0
        frame_cnt = 0
        frame_paths = []
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
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (self.short_side, np.Inf))
            
            frame = mmcv.imresize(frame, (new_w, new_h))
            frame_path = frame_tmp.format(frame_cnt + 1)
            frame_paths.append(frame_path)

            cv2.imwrite(frame_path, frame)
            frame_cnt += 1
            
        return dict(frame_paths= frame_paths,frame_name=frame_name, label=self.label)

    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.file_cnt >= self.num_video:
            raise StopIteration
        frame_data = self.read_video(self.file_cnt)
        self.file_cnt += 1
        return frame_data

    def __len__(self) -> int:
        return self.num_video

    def get_label(self) -> int:
        label_str = osp.basename(osp.splitext(self.video_path)[0])
        if label_str == "painting":
            label = 120
        else:
            label = 0
        assert label != 0
        return label

    def get_file_names(self) -> list:
        names = [x.split('.')[0] for x in self.video_list]
        return names

    def write_pkl(self, annotations: dict, name: str):
        dump(annotations, name)


class Annotation:
    def __init__(self, args) -> None:
        self.model = pm.PoseDetector(complexity=args.complexity)
        self.is_3D = args.dimension
        self.num_process = args.num_process
        self.anno_dict = {}
        self.test_scale = args.test_scale


    def skeleton_inference(self, frame_paths: list) -> list:
        ret = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frame = self.model.findPose(frame)
            if self.is_3D:
                point_dict = self.model.get3DPoints()
            else:  
                point_dict = self.model.getPoints()
            ret.append(point_dict)
        return np.array(ret)
        
 
    def get_anno(self, frame_data: dict) -> dict:
        keypoint = self.skeleton_inference(frame_data['frame_paths'])
        total_frames = len(frame_data['frame_paths'])
        
        return dict(frame_dir=frame_data['frame_name'], label=frame_data['label'], keypoint=keypoint, total_frames=total_frames)


    def get_anno_dict(self, file_control: FileController) -> dict:
        if self.num_process == 1:
            with tqdm(total=len(file_control)) as progress_bar:
                for frame_data in tqdm(file_control):
                        self.anno_dict[frame_data['frame_name']] = self.get_anno(frame_data)
                        progress_bar.update(1)
        else:
            pool = mp.Pool(self.num_process)
            annotations = pool.map(self.get_anno, file_control)
            pool.close()
            for anno in annotations:
                self.anno_dict[anno['frame_dir']] = anno

        return self.anno_dict

    def split_train_test(self, file_num: int, file_names: list, random_state=6897):
        file_names = [x for x in file_names if self.anno_dict is not None]
        file_names = np.array(file_names, dtype=str)
        test_size = int(file_num * self.test_scale)
        train_size = file_num - test_size

        np.random.seed(random_state)
        shuffled = np.random.permutation(file_num)
        file_names = file_names[shuffled]

        xsub_train = file_names[:train_size]
        xset_train = file_names[:train_size]
        xsub_val = file_names[train_size:]
        xset_val = file_names[train_size:]
        split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xset_train=xset_train, xset_val=xset_val)
        annotations = [self.anno_dict[name] for name in file_names]
        return dict(split=split, annotations=annotations)




def main():
    args = parse_args()
    videoFiles = FileController(args)
    annotator = Annotation(args)
    anno_dict = annotator.get_anno_dict(videoFiles)
    split_anno_dict = annotator.split_train_test(videoFiles.file_cnt, videoFiles.get_file_names())
    videoFiles.write_pkl(split_anno_dict, "custum_dataset.pkl")


    # tmp_frame_dir = osp.dirname('./tmp')
    # shutil.rmtree(tmp_frame_dir)


if __name__ == "__main__":
    main()


#python RGB2skeleton.py videos/painting