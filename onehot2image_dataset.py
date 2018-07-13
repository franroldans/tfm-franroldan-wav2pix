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
import csv
import scipy.io.wavfile as wavfile
import string
import random
import unicodedata
import h5py
import librosa
printable = set(string.printable)

class OneHot2YoutubersDataset(Dataset):

    def __init__(self, dataset_path, transform=None, split=None):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)
        self.transform = transform
        self.dataset = None
        #self.youtubers = [youtuber for youtuber in os.listdir(dataset_path)
        #                  if os.path.isdir(os.path.join(dataset_path, youtuber))]
        self.youtubers = self.read_youtubers('/imatge/froldan/PycharmProjects/youtubers_dataset/channels.csv')
        print(self.youtubers)
        self.num_youtubers = len(self.youtubers)
        self.split = split
        self.aux_path ='/imatge/froldan/mely_test'
        if split == 0:
            self.faces = pickle.load(open(config['youtubers_train_path'], 'rb'))
        elif split == 1:
            self.faces = pickle.load(open(config['youtubers_val_path'], 'rb'))
        elif split == 2:
            self.faces = pickle.load(open(config['youtubers_test_path'], 'rb'))
        elif split == 3:
            self.faces = pickle.load(open(config['youtubers_faces_path'], 'rb'))
            self.random_audios = os.listdir(self.aux_path)
        elif split == 4:
            self.faces = pickle.load(open(config['single_youtuber_path'], 'rb'))
        else:
            self.faces = pickle.load(open(config['youtubers_faces_path'], 'rb'))

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):

        path = self.faces[idx]
        format_path = self.format_filename(path)
        format_path = format_path.replace(" ", "") \
                .replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace('#', '') \
                .replace('&', '').replace(';', '').replace('!', '').replace(',', '').replace('$', '').replace('?', '')

        cropped_face = Image.open(filter(lambda x: x in printable, format_path).replace('.jpg', '.png'))
        if self.split is not 3:
            audio_path = format_path.replace("video", "audio").replace("cropped_frames", "frames").replace('.jpg', '.wav').replace('.png', '.wav')\
                .replace('cropped_face_frame', format_path.split('/')[7].replace('_cropped_frames', '') + '_preprocessed_frame')
            fm, wav_data = wavfile.read(filter(lambda x: x in printable, audio_path).replace('youtubers_audios_audios', 'youtubers_videos_audios').replace('.png', '.wav'))
            if fm != 16000:
                raise ValueError('Sampling rate is expected to be 16kHz!')
            youtuber = path.split('/')[5] #TODO: This is hardcoded, better change it!!!
            wav_data = self.abs_normalize_wave_minmax(wav_data)
            wav_data = self.pre_emphasize(wav_data)
            """y, sr = librosa.core.load(filter(lambda x: x in printable, audio_path).
                                      replace('youtubers_audios_audios', 'youtubers_videos_audios')
                                      .replace('.png', '.wav'), sr=16000)"""
            #mel_spec = librosa.feature.melspectrogram(y=wav_data, sr=16000, n_fft=512, hop_length=128)
            melspec_path = filter(lambda x: x in printable, audio_path).replace('youtubers_audios_audios', 'youtubers_videos_audios').replace('.wav', '.h5')
            #print(melspec_path)
            file_melspec = h5py.File(melspec_path, 'r')
            mel_spec = file_melspec['melspec'][:]
            file_melspec.close()
        else:
            audio_path = random.choice(self.random_audios)
            fm, wav_data = wavfile.read(os.path.join(self.aux_path, audio_path))
            if fm != 16000:
                raise ValueError('Sampling rate is expected to be 16kHz!')
            youtuber = path.split('/')[5]
            #youtuber = 'LuzuVlogs'
            wav_data = self.abs_normalize_wave_minmax(wav_data)
            wav_data = self.pre_emphasize(wav_data)
            mel_spec = np.array([1])
        onehot = self.youtubers.index(youtuber)
        wrong_face_path = self.get_dismatched_face(filter(lambda x: x in printable, audio_path).replace('youtubers_audios_audios', 'youtubers_videos_audios').replace('.png', '.wav'))
        wrong_face = Image.open(wrong_face_path)
        sample = {'onehot': self.to_categorical(onehot), 'face': np.array(cropped_face), 'audio': wav_data,
                  'audio_path': audio_path, 'mel_spec': mel_spec, 'wrong_face': np.array(wrong_face)}
        if self.transform:
            sample = self.transform(sample)
        sample['face'] = sample['face'].sub_(127.5).div_(127.5)
        sample['wrong_face'] = sample['wrong_face'].sub_(127.5).div_(127.5)
        return sample


    def to_categorical(self, token):
        """ 1-hot encodes a tensor """
        return np.eye(self.num_youtubers, dtype='uint8')[token]

    def read_youtubers(self, file_path):
        youtubers = []
        with open(file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                youtubers.append(row["Name"].replace(" ","").replace("'",""))
        return youtubers

    def abs_normalize_wave_minmax(self, wavdata):
        x = wavdata.astype(np.int32)
        imax = np.max(np.abs(x))
        x_n = x / imax
        return x_n

    def pre_emphasize(self, x, coef=0.95):
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat

    def format_filename(self, filename):
        try:
            filename = filename.decode('utf-8')
            s = ''.join((c for c in unicodedata.normalize('NFD', unicode(filename)) if unicodedata.category(c) != 'Mn'))
            return s.decode()
        except (UnicodeEncodeError, UnicodeDecodeError):
            return filename

    def get_dismatched_face(self, audio_path):
        selected_face = random.choice(self.faces)
        if selected_face.split('/')[5] == audio_path.split('/')[5]:
            selected_face = self.get_dismatched_face(audio_path)
        return selected_face


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
        onehot, face, audio, p, m , wf = sample['onehot'], sample['face'], sample['audio'], \
                                    sample['audio_path'], sample['mel_spec'], sample['wrong_face']
        img = transforms.ToPILImage()(face)
        wrong_img = transforms.ToPILImage()(wf)
        img = transforms.Scale((self.output_size, self.output_size))(img)
        wrong_img = transforms.Scale((self.output_size, self.output_size))(wrong_img)
        img = np.array(img, dtype=float)
        wrong_img = np.array(wrong_img, dtype=float)

        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = wrong_img
            rgb[:, :, 1] = wrong_img
            rgb[:, :, 2] = wrong_img
            wrong_img = rgb
        img = img.transpose(2, 0, 1) #Transpose image
        wrong_img = wrong_img.transpose(2, 0, 1) #Transpose image

        return {'onehot': torch.from_numpy(onehot).float(), 'face': torch.from_numpy(np.array(img)).float(),
                'audio': torch.from_numpy(audio).float(), 'audio_path': p, 'mel_spec': torch.from_numpy(m),
                'wrong_face': torch.from_numpy(wrong_img).float()}


"""#Testing implementation:
mydataset = OneHot2YoutubersDataset('/imatge/froldan/work/youtubers_videos_audios',
                                    transform=Rescale(64))
dataloader = DataLoader(mydataset, batch_size=64, shuffle=True)
data_iterator = iter(dataloader)
iterator = 0
while iterator < len(dataloader):
    sample = next(data_iterator)
    images = sample['face']
    onehot = sample['onehot']
    audio = sample['audio'].unsqueeze(1)
    melspec = sample['mel_spec']
    print(melspec.shape)
    print(audio.shape)
    iterator += 1
"""