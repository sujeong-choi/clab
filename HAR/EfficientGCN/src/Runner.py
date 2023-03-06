import logging, torch, os, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer
from . import dataset
from .dataset import Graph


action_names = ["others","painting","interview"]
class Runner(Initializer):
    def __init__(self, args) -> None:
        U.set_logging(args)
        self.args = args

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_save_dir()
        self.init_environment()
        self.init_device()
        self.init_data()
        self.init_model()
        # self.init_buffer()

    def init_buffer(self):
        self.buffer = np.zeros((3,15)) # 15개 label - 1분
        self.front, self.back = 0, 14

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)

    def init_data(self):
        self.videoFiles = dataset.FileController(self.args)
        self.skeletonMaker = dataset.SkeletonMaker(self.args)
        self.max_channel = 3
        self.max_frame = 144
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        
        graph = Graph(self.args.dataset)
        self.A = graph.A
        self.parts = graph.parts
        self.conn = graph.connect_joint
        T = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['num_frame']
        # inputs = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['inputs']
        self.data_shape = [3, 6, T, 25, 2]
        self.num_class = 20 #38 #121 # 2/6
        
        
    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def run(self):
        original_pd, updated_pd = [], []
        big3 = [0.0,0.0,0.0]
        big3_npy = None
        input_data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
        zero_arr =np.zeros((1, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        inputs = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['inputs']
        T = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['num_frame']
        self.previous_label = 0
        self.updated_label = 0

        self.model.eval()
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        first = True
        for idx, (frames, _) in tqdm(enumerate(self.videoFiles)):
            skeleton, _, _ = self.skeletonMaker.skeleton_inference(frames)
            
            skeleton_list = np.array([skeleton])
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton = skeleton_list

            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            input_data[:,:len(frames),:,:] = torch.from_numpy(skeleton.transpose(3, 1, 2, 0))
            
            data = [input_data[:,:T,:,:]]
            data = torch.tensor(data)
            data = data.type(torch.float64)
            x = data.float().to(self.device)            

            out, _ = self.model(x)
            out_prob = self.softmax(out[0].cpu().detach().numpy()) # 확률로 전환
            big3 = [max(out_prob[:17]), out_prob[17], out_prob[18]]

     
            reco_top1 = np.argmax(big3)
            if reco_top1!=self.updated_label and reco_top1==self.previous_label: 
                self.updated_label = reco_top1
            self.previous_label = reco_top1 

            original_pd.append(reco_top1)
            updated_pd.append(self.updated_label)
            top1_name = action_names[self.updated_label]
            res_str = "{} {:.2f}%".format(top1_name,big3[self.updated_label])

            self.videoFiles.write_videos(idx, top1_name, res_str, frames)
        

    def softmax(self,out):
        # print(f"out = {out}")
        exp_out = np.exp(out)
        sum_exp_put = np.sum(exp_out)
        y = (exp_out/ sum_exp_put) * 100
        return y
   