import logging, torch, os, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer
from . import dataset
from .dataset import Graph

# action_names = [
#             'drink water 1', 'eat meal/snack 2', 'brushing teeth 3', 'brushing hair 4', 'drop 5', 'pickup 6',
#             'throw 7', 'sitting down 8', 'standing up 9', 'clapping 10', 'reading 11', 'writing 12',
#             'tear up paper 13', 'wear jacket 14', 'take off jacket 15', 'wear a shoe 16', 'take off a shoe 17',
#             'wear on glasses 18','take off glasses 19', 'put on a hat/cap 20', 'take off a hat/cap 21', 'cheer up 22',
#             'hand waving 23', 'kicking something 24', 'put/take out sth 25', 'hopping 26', 'jump up 27',
#             'make a phone call 28', 'playing with a phone 29', 'typing on a keyboard 30',
#             'pointing to sth with finger 31', 'taking a selfie 32', 'check time (from watch) 33',
#             'rub two hands together 34', 'nod head/bow 35', 'shake head 36', 'wipe face 37', 'salute 38',
#             'put the palms together 39', 'cross hands in front 40', 'sneeze/cough 41', 'staggering 42', 'falling 43',
#             'touch head 44', 'touch chest 45', 'touch back 46', 'touch neck 47', 'nausea or vomiting condition 48',
#             'use a fan 49', 'punching 50', 'kicking other person 51', 'pushing other person 52',
#             'pat on back of other person 53', 'point finger at the other person 54', 'hugging other person 55',
#             'giving sth to other person 56', 'touch other person pocket 57', 'handshaking 58',
#             'walking towards each other 59', 'walking apart from each other 60' , 'put on headphone 61',
#             'take off headphone 62', 'shoot at the basket 63', 'bounce ball 64', 'tennis bat swing 65',
#             'juggling table tennis balls 66', 'hush (quite) 67', 'flick hair 68', 'thumb up 69',
#             'thumb down 70', 'make ok sign 71', 'make victory sign 72', 'staple book 73' , 'counting money 74',
#             'cutting nails 75', 'cutting paper (using scissors) 76', 'snapping fingers 77', 'open bottle 78',
#             'sniff (smell) 79', 'squat down 80', 'toss a coin 81', 'fold paper 82', 'ball up paper 83',
#             'play magic cube 84', 'apply cream on face 85', 'apply cream on hand back 86', 'put on bag 87',
#             'take off bag 88', 'put something into a bag 89', 'take something out of a bag 90', 'open a box 91',
#             'move heavy objects 92', 'shake fist 93', 'throw up cap/hat 94', 'hands up (both hands) 95',
#             'cross arms 96', 'arm circles 97', 'arm swings 98', 'running on the spot 99', 'butt kicks (kick backward) 100',
#             'cross toe touch 101', 'side kick 102', 'yawn 103', 'stretch oneself 104', 'blow nose 105',
#             'hit other person with something 106', 'wield knife towards other person 107', 
#             'knock over other person (hit with body) 108', 'grab other person’s stuff 109',
#             'shoot at other person with a gun 110', 'step on foot 111', 'high-five 112',
#             'cheers and drink 113', 'carry something with other person 114', 'take a photo of other person 115',
#             'follow other person 116', 'whisper in other person’s ear 117', 'exchange things with other person 118',
#             'support somebody with hand 119', 'finger-guessing game (playing rock-paper-scissors) 120', 'Painting! 121'
#         ]

action_names = [
            'writing 12-6', 
            'hand waving 23-9','pointing to sth with finger 31-10','rub two hands together 34-11',
            'nod head/bow 35-12', 'shake head 36-13','put the palms together 39-14', 'cross hands in front 40-15', 'touch head 44-16',
            'touch chest 45-17', 'touch back 46-18', 'touch neck 47-19', 'hush (quite) 67-20','thumb up 69-21', 
            'thumb down 70-22', 'make ok sign 71-23', 'make victory sign 72-24', 'staple book 73-25','cutting paper (using scissors) 76-26', 
            'snapping fingers 77-27', 'squat down 80-28', 'fold paper 82-29', 'ball up paper 83-30', 'hands up (both hands) 95-31',
            'cross arms 96-32', 'arm circles 97-33', 'arm swings 98-34','cross toe touch 101-35', 'stretch oneself 104-36',
            'high-five 112-37', 'Painting 121-38'
        ]
# action_names = ['painting','Staring','other thing'] # 2/6


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
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2

        sample_data = []
        sample_path = []
        data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
        zero_arr =np.zeros((1, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        
        graph = Graph(self.args.dataset)
        self.A = graph.A
        self.parts = graph.parts
        self.conn = graph.connect_joint

        T = self.args.dataset_args['ntu']['num_frame']
        inputs = self.args.dataset_args['ntu']['inputs']
        self.data_shape = [3, 6, T, 25, 2]
        self.num_class = 3 #38 #121 # 2/6
        
        for idx, (frames, _) in tqdm(enumerate(self.videoFiles)):
            skeleton, __ = self.skeletonMaker.skeleton_inference(frames)
            
            skeleton_list = np.array([skeleton])
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton_list = np.append(skeleton_list, zero_arr[:,:len(frames),:,:],axis=0)
            skeleton = skeleton_list

            # print(skeleton)

            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            data[:,:len(frames),:,:] = skeleton.transpose(3, 1, 2, 0)
            
            joint, velocity, bone = self.multi_input(data[:,:T,:,:])
            data_new = []
            if 'J' in inputs:
                data_new.append(joint)
            if 'V' in inputs:
                data_new.append(velocity)
            if 'B' in inputs:
                data_new.append(bone)
            data_new = np.stack(data_new, axis=0)

            sample_data.append([data_new])
            sample_path.append(self.videoFiles.video_list[idx])

        self.sample_data = np.array(sample_data)

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

    def run(self):
        self.model.eval()
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        for idx, data in tqdm(enumerate(self.sample_data)):

            data = torch.tensor(data)
            data = data.type(torch.float64)
            x = data.float().to(self.device)

            out, _ = self.model(x)
            # print(out)
            reco_top1 = out.max(1)[1]
            print("\ntop1=",action_names[reco_top1])
            print("(",out.max(1)[0],")\n")
            
            reco_top3 = torch.topk(out,3)[1]
            
            top1_name = action_names[reco_top1]
            top3_names = [action_names[idx] for idx in reco_top3[0]]

            self.videoFiles.write_videos(idx, top1_name, top3_names)
        

        
        
