import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train',repeat_num = 10, transform=None):
        self.root = root
        self.phase = phase
        self.labels = {}
        self.transform = transform
        self.repeat_num = repeat_num
        if self.phase != 'train':
            self.label_path = os.path.join(root, self.phase, self.phase + '_label.csv')
            # used to prepare the labels and images path
            self.direc_df = pd.read_csv(self.label_path)
            self.direc_df.columns = ["image1", "image2", "label"]
            self.dir = os.path.join(root, self.phase)
        else:
            self.train_meta_dir = os.path.join(root, self.phase, self.phase + '_meta.csv')
            train_meta = pd.read_csv(self.train_meta_dir)

            train_label_file = os.path.join(root, self.phase, f'{self.phase}_label_{self.repeat_num}.csv')
            print('train_label_file:', train_label_file, os.path.isfile(train_label_file))

            if not os.path.isfile(train_label_file):
                train_data = []
                # make_true_pair
                id_list = list(set(train_meta['face_id']))
                for i in range(int(self.repeat_num)):
                    if i % 10 == 0:
                        print(i)
                    for id in id_list:
                        pair = []
                        candidate = train_meta[train_meta['face_id'] == int(id)]
                        pair.append(candidate[candidate['cam_angle']=='front'].sample(1, random_state = i)['file_name'].item())
                        pair.append(candidate[candidate['cam_angle']=='side'].sample(1, random_state = i)['file_name'].item())
                        pair.append(0)
                        train_data.append(pair)
                # make_false_pair
                id_list = list(set(train_meta['face_id']))
                for i in range(int(self.repeat_num)):
                    if i % 10 == 0:
                        print(i)
                    for id in id_list:
                        pair = []
                        candidate = train_meta[train_meta['face_id'] == int(id)]
                        candidate_others = train_meta[train_meta['face_id'] != int(id)]
                        pair.append(candidate[candidate['cam_angle']=='front'].sample(1, random_state = i)['file_name'].item())
                        pair.append(candidate_others[candidate_others['cam_angle']=='side'].sample(1, random_state = i)['file_name'].item())
                        pair.append(1)
                        train_data.append(pair)
            else:
                train_data = pd.read_csv(train_label_file)
            self.direc_df = pd.DataFrame(train_data)
            self.direc_df.columns = ["image1", "image2", "label"]
            self.dir = os.path.join(root, self.phase)
            #self.direc_df.to_csv(os.path.join(root, self.phase, self.phase + '_label.csv'), mode='w', index=False)
            #self.label_path = os.path.join(root, self.phase, self.phase + '_label.csv')
            self.direc_df.to_csv(os.path.join(root, self.phase, f'{self.phase}_label_{self.repeat_num}.csv'), mode='w', index=False)
            self.label_path = os.path.join(root, self.phase, f'{self.phase}_label_{self.repeat_num}.csv')

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.dir, self.direc_df.iat[index, 0])
        image2_path = os.path.join(self.dir, self.direc_df.iat[index, 1])
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        if self.phase != 'test':
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1,
                    torch.from_numpy(np.array([int(self.direc_df.iat[index, 2])], dtype=np.float32)))
        elif self.phase == 'test':
            dummy = ""
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1, dummy)

    def __len__(self):
        return len(self.direc_df)

    def get_label_file(self):
        print('label_file:', self.label_path)
        return self.label_path

def data_loader(root, phase='train', batch_size=64,repeat_num=10):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = CustomDataset(root, phase, repeat_num,
                            transform=transforms.Compose([transforms.Resize((105,105)),
                                                                      transforms.ToTensor()
                                                                      ]))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()