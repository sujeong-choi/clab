import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization

GAP = 2

class NTU_Reader():
    def __init__(self, args, root_folder, transform, ntu60_path, ntu120_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        self.used_actions = [5,6,7,8,9,12,13,23,25,31,34,35,36,39,40,44,45,46,47,67,69,70,71,72,73,76,77,80,82,83,95,96,97,98,101,104,112,121]
        self.C2O, self.O2C = dict(), dict()
        for i in range(len(self.used_actions)):
            self.C2O[i+1]=self.used_actions[i]
            self.O2C[self.used_actions[i]]=i+1

        # Set paths
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))

        training_samples['ntu-xsub38']=training_samples['ntu-xsub120'] # training : count = 20322 / testing : count = 16484

        '''
        all = {0: 15, 1: 4, 2: 10, 3: 11, 4: 15, 5: 1, 6: 2, 7: 15, 8: 10, 9: 15, 10: 4, 11: 6, 12: 15, 13: 4, 14: 14, 15: 15, 16: 15, 17: 2, 18: 15, 19: 15, 20: 15, 21: 15, 22: 10, 23: 15, 24: 7, 25: 15, 26: 15, 27: 15, 28: 15, 29: 15, 30: 15, 31: 15, 32: 15, 33: 15, 34: 15, 35: 15, 36: 15, 37: 15, 38: 15, 39: 15, 40: 15, 41: 15, 42: 15, 43: 15, 44: 15, 45: 15, 46: 15, 47: 15, 48: 15, 49: 15, 50: 15, 51: 15, 52: 15, 53: 15, 54: 15, 55: 15, 56: 15, 57: 15, 58: 15, 59: 15, 60: 15, 61: 15, 62: 15, 63: 15, 64: 15, 65: 15, 66: 15, 67: 15, 68: 15, 69: 15, 70: 15, 71: 15, 72: 15, 73: 15, 74: 15, 75: 15, 76: 15, 77: 15, 78: 15, 79: 15, 80: 15, 81: 15, 82: 15, 83: 15, 84: 15, 85: 15, 86: 15, 87: 15, 88: 15, 89: 15, 90: 15, 91: 15, 92: 15, 93: 15, 94: 15, 95: 15, 96: 15, 97: 15, 98: 15, 99: 15, 100: 15, 101: 15, 102: 15, 103: 15, 104: 3, 105: 15, 106: 1, 107: 2, 108: 13, 109: 15, 110: 15, 111: 15, 112: 1, 113: 15, 114: 15, 115: 15, 116: 15, 117: 15, 118: 15, 119: 15, 120: 10, 121: 14}
        painting = {0: 15, 1: 4, 2: 10, 3: 11, 4: 15, 5: 1, 6: 2, 7: 15, 8: 10, 9: 15, 10: 4, 11: 6, 12: 15, 13: 4, 14: 14, 15: 15, 16: 15, 17: 2, 18: 15, 19: 15, 20: 15, 21: 15, 22: 10, 23: 15, 24: 7, 25: 15, 26: 15, 27: 15, 28: 15, 29: 15, 30: 15, 31: 15, 32: 15, 33: 15, 34: 15, 35: 15, 36: 15, 37: 15, 38: 15, 39: 15, 40: 15, 41: 15, 42: 15, 43: 15, 44: 15, 45: 15, 46: 15, 47: 15, 48: 15, 49: 15, 50: 15, 51: 15, 52: 15, 53: 15, 54: 15, 55: 15, 56: 15, 57: 15, 58: 15, 59: 15, 60: 15, 61: 15, 62: 15, 63: 15, 64: 15, 65: 15, 66: 15, 67: 15, 68: 15, 69: 15, 70: 15, 71: 15, 72: 15, 73: 15, 74: 15, 75: 15, 76: 15, 77: 15, 78: 15, 79: 15, 80: 15, 81: 15, 82: 15, 83: 15, 84: 15, 85: 15, 86: 15, 87: 15, 88: 15, 89: 15, 90: 15, 91: 15, 92: 15, 93: 15, 94: 15, 95: 15, 96: 15, 97: 15, 98: 15, 99: 15, 100: 15, 101: 15, 102: 15, 103: 15, 104: 3, 105: 15, 106: 1, 107: 2, 108: 13, 109: 15, 110: 15, 111: 15, 112: 1, 113: 15, 114: 15, 115: 15, 116: 15, 117: 15, 118: 15, 119: 15, 120: 10, 121: 14}
        '''
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        try:
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        except:
            logging.info('')
            logging.error('Error: Wrong in loading ignored sample file {}'.format(ntu_ignored))
            raise ValueError()

        # Get skeleton file list
        self.file_list = []
        check = dict()
        for folder in [ntu60_path, ntu120_path]:
            for filename in os.listdir(folder):
                if '38' in self.dataset:
                    action_loc = filename.find('A')
                    action_class = int(filename[(action_loc+1):(action_loc+4)])

                    if action_class in self.used_actions:
                        self.file_list.append((folder, filename))
                else:   self.file_list.append((folder, filename))
            if ('120' not in self.dataset) and ('38' not in self.dataset):  # for NTU 60, only one folder
                break


    def read_file(self, file_path):
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline()) # 원래 총 frame 수
            gap = 1
            # fps 통일
            if '121' not in file_path:
                gap = GAP
                frame_num = int(frame_num/gap)
            

            # print(f"원래 frame 수 ={ori_frame_num}  >>>> 바뀐 frame 수 = {frame_num}\n")

            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    person_info = fr.readline().strip().split() # person ID
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
                # gap-1 개의 frame 건너뜀.
                for _ in range((gap-1)):  
                    person_num = int(fr.readline())
                    for j in range(person_num*27):
                        fr.readline()
        return skeleton[:,:frame_num,:,:], frame_num

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def gendata(self, phase):
        # count = 0
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        check = dict()
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        # print(f"iterizer = {len(iterizer)}")
        for folder, filename in iterizer:
            if filename in self.ignored_samples:
                continue

            # Get sample information
            file_path = os.path.join(folder, filename)
            setup_loc = filename.find('S')
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            action_class = self.O2C[int(filename[(action_loc+1):(action_loc+4)])]



            # Distinguish train or eval sample
            if self.dataset == 'ntu-xview':
                is_training_sample = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120' or self.dataset=='ntu-xsub38':
                is_training_sample = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120':
                is_training_sample = (setup_id in self.training_sample)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            skeleton, frame_num = self.read_file(file_path)
            
            # Select person by max energy
            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)

            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)
            if subject_id not in check : check[subject_id]=1
            else : check[subject_id]+=1
        #     count+=1
        # print(f"count = {count}\n")

        # Save label
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
