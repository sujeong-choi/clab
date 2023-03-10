import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization

GAP = 2

class NTU_Reader():
    def __init__(self, args, root_folder, transform,  mediapipe_path, **kwargs): # ntu60_path, ntu120_path, 제거 2/9
        self.max_channel = 3
        self.max_frame = 144
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        self.used_actions = [1,4,5,6,10,18,19,28,29,33,49,68,78,91,96,103,104,121,122]
        self.C2O, self.O2C = dict(), dict()
        for i in range(len(self.used_actions)):
            self.C2O[i]=self.used_actions[i]
            self.O2C[self.used_actions[i]]=i

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
            1, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 24,25,26, 27, 28, 31,32, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]
        training_samples['ntu-xsub120'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))

        training_samples['ntu-xset38']=training_samples['ntu-xset120'] # training : count = 20,357 / testing : count = 17,401

        training_samples['mediapipe-xset'] = training_samples['ntu-xset120'] # 2/6
        training_samples['mediapipe-ntu-xsub'] = training_samples['ntu-xsub'] # 2/9
        training_samples['mediapipe-ntu-xset'] = [1,2,4,6,8,10,11,12,14,16,18,20,21,22,24,26,28,30,31, 32] # 2/11

        # print(f"self.dataset = {self.dataset}")
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
        for folder in [mediapipe_path]:
            for filename in os.listdir(folder):
                if 'mediapipe' in self.dataset:
                    action_loc = filename.find('A')
                    action_class = int(filename[(action_loc+1):(action_loc+4)])

                    if action_class in self.used_actions:
                        self.file_list.append((folder, filename))
                else:   self.file_list.append((folder, filename))
            if ('120' not in self.dataset) and ('38' not in self.dataset):  # for NTU 60, only one folder
                break

    # Function to read the created skeleton file
    def read_file(self, file_path):
        original_skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32) #(4,300,25,3)
        minus_skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32) #(4,300,25,3)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline()) # 원래 총 frame 수, Total number of frames originally(it will be 60 in our case)

            
            for frame in range(frame_num):
                person_num = int(fr.readline()) #it will be 1
                for person in range(person_num): #for each person
                    _ = fr.readline().strip().split() # person ID(ignored)
                    joint_num = int(fr.readline()) #it will be 25
                    for joint in range(joint_num): #for each joint(keypoints)
                        joint_info = fr.readline().strip().split() # extract x,y,z coordinates (ex. [0.231231, -0.23213, 0.1234])
                        # print(f"joint_info = {joint_info}")
                        minus_joint_info = [joint_info[0],joint_info[1], str(-float(joint_info[2]))]
                        # print(f"minus_joint_info = {minus_joint_info}")
                        original_skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32) #add x,y,z data to skeleton list
                        minus_skeleton[person,frame,joint,:] = np.array(minus_joint_info[:self.max_channel], dtype=np.float32) #add x,y,z data to skeleton list
        return original_skeleton[:,:frame_num,:,:],minus_skeleton[:,:frame_num,:,:], frame_num    #return skeleton data & frame_num

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def gendata(self, phase):
        count = 0
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
            # print(f" A = {int(filename[(action_loc+1):(action_loc+4)])}\n")
            action_class = self.O2C[int(filename[(action_loc+1):(action_loc+4)])]
            # action_class = int(filename[(action_loc+1):(action_loc+4)])


            # Distinguish train or eval sample
            if self.dataset == 'ntu-xview':
                is_training_sample = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120' or self.dataset == "mediapipe-ntu-xsub":
                is_training_sample = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120' or self.dataset=='ntu-xset38' or self.dataset=='mediapipe-xset'  or self.dataset == "mediapipe-ntu-xset":
                is_training_sample = (setup_id in self.training_sample)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32) #(3,300,25,2)
            original_skeleton, minus_skeleton, frame_num = self.read_file(file_path)
            
            # Select person by max energy
            #(Actually, it's a meaningless code because there's only one person,
            # but I used this code when I was learning the model, so I used it to match the format.)
            skeleton = original_skeleton
            for i in range(2):
                energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)]) 
                index = energy.argsort()[::-1][:self.select_person_num]
                skeleton = skeleton[index]
                data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0) 

                sample_data.append(data)
                sample_path.append(file_path)
                sample_label.append(action_class)  # to 0-indexed
                sample_length.append(frame_num)

                if subject_id not in check : check[subject_id]=1
                else : check[subject_id]+=1
                count+=1

                if action_class == 18:
                    break
                
                skeleton = minus_skeleton
        print(f"count = {count}\n")

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