import os
import torch
from torch.utils.data import Dataset

from common import loadWAV, AugmentWAV, PreEmphasis


class SEEDEmbeddingDataset(Dataset):
    def __init__(self, speaker_model, train_list, train_path, max_frames, musan_path=None, rir_path=None):

        self.speaker_model = speaker_model
        self.speaker_model.eval()
        self.max_frames  = max_frames
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        self.data_list  = []

        for line in train_list:
            data = line.strip().split()
            filename = os.path.join(train_path, data[1])
            self.data_list.append(filename)

    def __getitem__(self, index):
        
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False) 

        audio_reverb = self.augment_wav.reverberate(audio)
        audio_noise  = self.augment_wav.additive_noise("noise", audio)
        audio_music  = self.augment_wav.additive_noise("music", audio)

        audio_tensor        = torch.FloatTensor(audio)
        audio_reverb_tensor = torch.FloatTensor(audio_reverb)
        audio_noise_tensor  = torch.FloatTensor(audio_noise)
        audio_music_tensor  = torch.FloatTensor(audio_music)
        
        all_audio = torch.stack([
            audio_tensor, audio_reverb_tensor, 
            audio_noise_tensor, audio_music_tensor
        ])
        with torch.no_grad():
            embeddings = self.speaker_model(all_audio)
            
            x_clean = embeddings[0]
            y_noisy = embeddings[1:]
        
        return {
            "clean": x_clean.cpu(),
            "noisy": y_noisy.cpu()
        }

    def __len__(self):
        return len(self.data_list)
    
class test_dataset_loader(Dataset):
    def __init__(self, test_list, max_frames, test_path):

        self.max_frames  = max_frames

        dictkeys = list(set([x.split()[0] for x in test_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(test_list):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(test_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=1)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)