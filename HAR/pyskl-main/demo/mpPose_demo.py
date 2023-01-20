# 최신
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
import time
import PoseModule as pm

from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer
from pyskl.utils.visualize import Vis3DPose


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/pyskl/ckpt/'
                 'posec3d/slowonly_r50_ntu120_xsub/joint.pth'),
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt', # label 지정 부분
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='frame rate desired for processing')
    args = parser.parse_args()
    return args

# - 입력 : 비디오 경로 
# - do : 비디오 -> frames와 frame_paths(img_%d.jpg) 생성(저장) 후 반환
def frame_extraction( video_path, short_side, dst_fps):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    skip_frame = float(fps / dst_fps)
    frame_cnt=0
    frames = []
    frame_paths = []
    # flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    prev_time = 0
    FPS = 6
    while True:
        flag, frame = vid.read()
        if not flag:
            break
        if frame_cnt <skip_frame:
            frame_cnt += 1
            continue
        frame_cnt = 0
        # current_time = time.time()-prev_time
        # if (flag is True) and (current_time > 1./ FPS) :
        #     prev_time = time.time()

        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1

    return frame_paths, frames


def skeleton_inference(config,frame_paths):
    model = pm.PoseDetector()
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    ret =[]
    first = True
    for file in frame_paths:
        image = cv2.imread(file)
        image = model.findPose(image, False)
        for x in config.data.test.pipeline:
            if x['type'] == 'PreNormalize3D': 
                point_dict = model.get3DPoints(image)
                break
            else :  
                point_dict = model.getPoints(image)
                break
        # point_dict = model.getWorldPoints(image)
        ret.append(point_dict)
        prog_bar.update()
        if first : 
            print(ret)
            first=False
    return ret

    # ret = [[{'keypoints':array([[x,y,score]1,..,[x,y,score]k])}]1,..,[]m] / m : frame의 개수 , k : keypoint 개수
    # [[(x,y,score)1,..,(x,y,score)]1,..,[]m]
            

def main():
    start_time = time.time()
    args = parse_args()
    print(args.video)

    frame_paths, original_frames = frame_extraction(args.video, args.short_side, args.fps) # frame들과 그에대한 path들 반환
    num_frame = len(frame_paths) # 프레임의 수
    print(f"num_frame = {num_frame}")
    h, w, _ = original_frames[0].shape# 프레임의 높이, 너비, _

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]


    model = init_recognizer(config, args.checkpoint, args.device) 

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map, encoding='UTF-8').readlines()]

    # Get Human detection results

    pose_results = skeleton_inference(config,frame_paths)
    # print(pose_results)
    torch.cuda.empty_cache()
    # print(pose_results)
    # pose_results : 여러 frame의 skeleton 정보들
    # poses : 한 frame에서 여러 사람의 skeleton 정보들
    # pose : 한 frame에서 한 사람의 skeleton 정보
    
    skeleton_end_time = time.time()
    
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    num_person = max([len(x) for x in pose_results]) # 한 프레임에서 최대 사람 수 
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
    num_keypoint = 25 # 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 3),
                        dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :]
            #keypoint_score[j, i] = pose[:, 2]
            
    fake_anno['keypoint'] = keypoint[:]
    vis3D = Vis3DPose(fake_anno,angle=(90, 90), fig_size=(16,16))
    vid = vis3D.vis()
    results = inference_recognizer(model, fake_anno)
    # results[0] = dict[tuple(str, float)]: Top-5 recognition result dict.
    action_label = label_map[results[0][0]] # action label 지정
    print("action label: ", action_label)
                                 
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)
    finish_time = time.time()
    print("skeleton 추출 시간(sec): {}".format(skeleton_end_time-start_time))
    print("pose 추론 시간(sec): {}".format(finish_time-skeleton_end_time))
    print("전체 추론 시간(sec): {}".format(finish_time-start_time))
    


if __name__ == '__main__':
    main()


# 2D : python demo/mpPose_demo.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
# python demo/mpPose_demo.py data/test/self_data_(8).mp4 data/output/self_data_(8).mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
# 3D : python demo/mpPose_demo.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_3dkp/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth
# python demo/mpPose_demo.py data/test/KakaoTalk_20230118_204203127.mp4  data/output/3D.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_3dkp/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth

