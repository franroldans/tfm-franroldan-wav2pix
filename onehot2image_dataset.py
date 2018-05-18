import os
import io
from torch.utils.data import Dataset, DataLoader
import cPickle as pickle
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import yaml

class OneHot2YoutubersDataset(Dataset):

    def __init__(self, dataset_path, transform=None, split=None):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)
        self.transform = transform
        self.dataset = None
        self.youtubers = [youtuber for youtuber in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, youtuber))]
        self.num_youtubers = len(self.youtubers)
        if split == 0:
            self.faces = pickle.load(open(config['youtubers_train_path'], 'rb'))
        elif split == 1:
            self.faces = pickle.load(open(config['youtubers_val_path'], 'rb'))
        elif split == 2:
            self.faces = pickle.load(open(config['youtubers_test_path'], 'rb'))
        else:
            self.faces = pickle.load(open(config['youtubers_faces_path'], 'rb'))

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        path = self.faces[idx]
        cropped_face = Image.open(path)
        youtuber = path.split('/')[5] #TODO: This is hardcoded, better change it!!!
        onehot = self.youtubers.index(youtuber)
        sample = {'onehot': self.to_categorical(onehot), 'face': np.array(cropped_face)}
        if self.transform:
            sample = self.transform(sample)
        sample['face'] = sample['face'].sub_(127.5).div_(127.5)
        return sample

    def to_categorical(self, token):
        """ 1-hot encodes a tensor """
        return np.eye(self.num_youtubers, dtype='uint8')[token]

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If int, output generated is an
        square image (output_size, output_size, channels). If tuple, output matches with
        output_size (output_size[0], output_size[1], channels).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        onehot, face = sample['onehot'], sample['face']
        img = transforms.ToPILImage()(face)
        img = transforms.Scale((self.output_size, self.output_size))(img)
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        img = img.transpose(2, 0, 1)
        #return {'onehot': torch.LongTensor([onehot]), 'face': torch.from_numpy(np.array(img)).float()}
        return {'onehot': torch.from_numpy(onehot).float(), 'face': torch.from_numpy(np.array(img)).float()}

#Testing implementation:
mydataset = OneHot2YoutubersDataset('/imatge/froldan/work/youtubers_videos_audios',
                                    transform=Rescale(64))
dataloader = DataLoader(mydataset, batch_size=64, shuffle=True)
data_iterator = iter(dataloader)
sample = next(data_iterator)
images = sample['face']
onehot = sample['onehot']
print(images.shape)