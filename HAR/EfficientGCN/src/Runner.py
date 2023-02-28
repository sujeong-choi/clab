import logging, torch, os, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer
from . import dataset
from .dataset import Graph

# action_names = [
#             'drink water 1',  'brushing hair 4', 'drop 5', 'pickup 6',
#             'clapping 10', 'writing 12',
#             'wear on glasses 18','take off glasses 19', 'make a phone call 28', 'playing with a phone 29',
#             'check time (from watch) 33',
#             'use a fan 49', 'flick hair 68', 'open bottle 78',
#             'open a box 91',
#             'cross arms 96', 'yawn 103', 'stretch oneself 104', 'Painting 121', 'interview 122'
#         ]

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
        self.init_buffer()

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

        # sample_data = []
        # sample_path = []
        # data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
        # zero_arr =np.zeros((1, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        
        graph = Graph(self.args.dataset)
        self.A = graph.A
        self.parts = graph.parts
        self.conn = graph.connect_joint
        T = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['num_frame']
        # inputs = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['inputs']
        self.data_shape = [3, 6, T, 25, 2]
        self.num_class = 20 #38 #121 # 2/6
        
        # for idx, (frames, _) in tqdm(enumerate(self.videoFiles)):
        #     skeleton, _, _ = self.skeletonMaker.skeleton_inference(frames)
            
        #     skeleton_list = np.array([skeleton])
        #     skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
        #     skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
        #     skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
        #     skeleton = skeleton_list

        #     # print(skeleton)

        #     energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
        #     index = energy.argsort()[::-1][:self.select_person_num]
        #     skeleton = skeleton[index]
        #     data[:,:len(frames),:,:] = skeleton.transpose(3, 1, 2, 0)
            
        #     joint, velocity, bone = self.multi_input(data[:,:T,:,:])
        #     data_new = []
        #     if 'J' in inputs:
        #         data_new.append(joint)
        #     if 'V' in inputs:
        #         data_new.append(velocity)
        #     if 'B' in inputs:
        #         data_new.append(bone)
        #     data_new = np.stack(data_new, axis=0)

        #     sample_data.append([data_new])
        #     sample_path.append(self.videoFiles.video_list[idx])

        # self.sample_data = np.array(sample_data)

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        return joint, velocity, bone

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def get_buffer_prob(self):
        start , end = self.front, self.back
        total_probability = [0.0,0.0,0.0]
        size = 15
        alpha = 1/3 # 1: 1/4 2: 1/3

        for j in range(0,3):
            start = self.front
            for i in range(size):
                if start == end: break
                total_probability[j]+=self.buffer[j][start]*((alpha)**(size-i))
                start = int((start+1)%size)
        
        return total_probability

    def update_queue(self,prob):
        size = 15
        for j in range(0,3):
            self.buffer[j][self.front]=prob[j]
        self.front, self.back = int((self.front+1)%size) , int((self.back+1)%size)


    def run(self):
        pd = []
        big3 = [0.0,0.0,0.0]
        big3_npy = None
        input_data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
        zero_arr =np.zeros((1, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        inputs = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['inputs']
        T = self.args.dataset_args[list(self.args.dataset_args.keys())[0]]['num_frame']

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
            
            joint, velocity, bone = self.multi_input(input_data[:,:T,:,:])
            data_new = []
            if 'J' in inputs:
                data_new.append(joint)
            if 'V' in inputs:
                data_new.append(velocity)
            if 'B' in inputs:
                data_new.append(bone)
            data_new = np.stack(data_new, axis=0)
            data = [data_new]

        # for idx, data in tqdm(enumerate(self.sample_data)):

            data = torch.tensor(data)
            data = data.type(torch.float64)
            x = data.float().to(self.device)

            out, _ = self.model(x)
            out_prob = self.softmax(out[0].cpu().detach().numpy()) # 확률로 전환
            big3 = [sum(out_prob[:18]), out_prob[18], out_prob[19]]
            if first :
                big3_npy = np.array(big3)
            else:
                big3_npy = np.append(big3_npy,big3)
            # res_str = "\n\nbefore -> Prob: painting:{0:.2f}%, interview:{1:.2f}%, others:{2:.2f}%".format(big3[1],big3[2],big3[0])
            # print(res_str)
            # big3 = [big3[j]+self.get_buffer_prob()[j] for j in range(0,3)] # 이전 확률값 반영
            # res_str = "tmmp -> Prob: painting:{0:.2f}%, interview:{1:.2f}%, others:{2:.2f}%\n".format(big3[1],big3[2],big3[0])
            # print(res_str)
            # big3_prob = big3/sum(big3)*100
            # self.update_queue(big3_prob) # buffer 업데이트
            # print(f"buffer = {self.buffer}")
            reco_top1 = np.argmax(big3)
            pd.append(reco_top1)
            top1_name = action_names[reco_top1]
            res_str = "painting:{0:.2f}%, interview:{1:.2f}%, others:{2:.2f}%".format(big3[1],big3[2],big3[0])
            # print(res_str)
            # print(out)
            # reco_top1 = out.max(1)[1]
            # res_str = self.print_softmax(out)
            # top1_name = action_names[reco_top1]

            self.videoFiles.write_videos(idx, top1_name, res_str, frames)
        
        #save all prob
        # self.get_FNFP(pd)
        # print(f"pd={pd}")
        np.save('./har_npy',big3_npy)

    def softmax(self,out):
        # print(f"out = {out}")
        exp_out = np.exp(out)
        sum_exp_put = np.sum(exp_out)
        y = (exp_out/ sum_exp_put) * 100
        return y

    
    def get_FNFP(self,pred):
        gt = [0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        0,0,0,0,0,2,2,2,2,2,
                        2,2,2,2,2,2,2,2,2,2,
                        2,2,2,2,2,2,2,2,2,2,
                        2,2,2,2,2,2,2,2,2,2,
                        2,2,2,2,2,2,2,2,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,0,0,0,
                        0,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,0,0,0,1,
                        1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,0,2,2]
        size = len(pred)
        FN, FP = [0,0,0],[0,0,0]
        accuracy = 0.0
        F = 0

        for i in range(size):

            if gt[i]==0 and pred[i] != 0 : FN[0] +=1
            elif gt[i]!=0 and pred[i] ==0 : FP[0] +=1

            # FN : painting인데 다른 label을 낸 경우
            if gt[i]==1 and pred[i] != 1 : FN[1] +=1
            # FP : painting이 아닌데 painting이라고 한 경우
            elif gt[i]!=1 and pred[i] ==1 : FP[1] +=1

            if gt[i]==2 and pred[i] != 2 : FN[2] +=1
            elif gt[i]!=2 and pred[i] ==2 : FP[2] +=1

            if gt[i] != pred[i] : F +=1
        print(f"==========other==========\n ")
        print(f"FN 비율= {float(FN[0])/size*100} %")
        print(f"FP 비율= {float(FP[0])/size*100} %")
        print(f"==========painting==========\n ")
        print(f"FN 비율= {float(FN[1])/size*100} %")
        print(f"FP 비율= {float(FP[1])/size*100} %")
        print(f"==========interview==========\n ")
        print(f"FN 비율= {float(FN[2])/size*100} %")
        print(f"FP 비율= {float(FP[2])/size*100} %")
        print(f"accuracy = {100.0-float(F)/size*100} %")
    # def print_softmax(self, out: list):
        # res_str = "Prob: painting:{0:.2f}%, interview:{1:.2f}%, others:{2:.2f}%".format(out[18], out[19], sum(out[0:17]))
        # return res_str

        
        
