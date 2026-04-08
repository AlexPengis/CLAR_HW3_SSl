import matplotlib.pyplot as plt
from dataset.AudioMNIST import AudioMnistDataset
from typing import Tuple, Optional, Union
from torch import Tensor
import os
import torch
import torchaudio
import torchaudio.transforms as transforms
import math
import random

try:
    from dataset.speechcommands import SPEECHCOMMANDS as _SCBase
except ImportError:
    _SCBase = object  # torchaudio version doesn't support SpeechCommands download utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSC(_SCBase):
    def __init__(self, subset: str = None, percentage = 1, batch_size = 256):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        def masklabel( percentage = 100, batch_size=256, trainsize = 0):
            if percentage == 100:
                return None

            random.seed(0) # Important!            
            index_ = [random.sample(range(i, i+batch_size), int(batch_size*(1-(percentage/100))) ) for i in range(0, trainsize, batch_size) ]
            flattened_list = [item for sublist in index_ for item in sublist]
            return flattened_list

        self.subset = subset

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            
            self.index_list = masklabel(percentage=percentage, batch_size=batch_size, trainsize=len(self._walker))


#labels of the dataset, (35)
labels =  ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow','forward','four',
           'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on','one', 'right',
           'seven', 'sheila', 'six', 'stop', 'three','tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero', 'None']

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)

    targets = torch.stack(targets)



    return tensors, targets


# ── AudioMNIST ────────────────────────────────────────────────────────────────

class SubsetAudioMNIST(AudioMnistDataset):
    """AudioMNIST subset with optional semi-supervised label masking.

    Args:
        root (str): Path to the AudioMNIST root directory
            (the folder that contains the ``data/`` subdirectory).
        subset (str or None): "training", "validation", "testing", or None.
        percentage (int): Percentage of labelled samples to keep (1-100).
            100 means fully supervised (no masking).
        batch_size (int): Batch size used to compute the masking index.
    """
    def __init__(self, root: str, subset: str = None, percentage: int = 100, batch_size: int = 256):
        super().__init__(root, subset=subset)

        def masklabel(percentage=100, batch_size=256, trainsize=0):
            if percentage == 100:
                return None
            random.seed(0)  # Important!
            index_ = [random.sample(range(i, i + batch_size), int(batch_size * (1 - (percentage / 100))))
                      for i in range(0, trainsize, batch_size)]
            return [item for sublist in index_ for item in sublist]

        if subset == "training":
            self.index_list = masklabel(percentage=percentage, batch_size=batch_size, trainsize=len(self._walker))


# AudioMNIST labels: digits 0-9, index 10 = "None" (masked label)
audiomnist_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'None']

def label_to_index_audiomnist(word):
    return torch.tensor(audiomnist_labels.index(word))

def index_to_label_audiomnist(index):
    return audiomnist_labels[index]

_AUDIOMNIST_RESAMPLE = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
_AUDIOMNIST_TARGET_LEN = 16000  # 1 second at 16 kHz, matches STFT/Mel config in supervised.py

def collate_fn_audiomnist(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        # Resample from 48 kHz (AudioMNIST native) to 16 kHz (model input rate)
        waveform = _AUDIOMNIST_RESAMPLE(waveform)
        # Crop or zero-pad to a fixed 16000 samples so all batches have the same shape
        if waveform.shape[-1] < _AUDIOMNIST_TARGET_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, _AUDIOMNIST_TARGET_LEN - waveform.shape[-1]))
        else:
            waveform = waveform[..., :_AUDIOMNIST_TARGET_LEN]
        tensors.append(waveform)
        targets.append(label_to_index_audiomnist(label))
    tensors = torch.stack(tensors)   # [B, 1, 16000] — constant shape, no padding mask needed
    targets = torch.stack(targets)
    return tensors, targets


def getDataAudioMNIST(root: str = "./AudioMNIST", batch_size: int = 32,
                      num_workers: int = 0, pin_memory: bool = False, percentage: int = 100):
    train_set = SubsetAudioMNIST(root, subset="training",   percentage=percentage, batch_size=batch_size)
    val_set   = SubsetAudioMNIST(root, subset="validation", percentage=percentage)
    test_set  = SubsetAudioMNIST(root, subset="testing",    percentage=percentage)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,  collate_fn=collate_fn_audiomnist, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_audiomnist, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_audiomnist, num_workers=0)

    return train_loader, test_loader, val_loader


def createSpectograms(audio, stft, mel_transform):
    
    # Create magnitude and phase
    mag = stft(audio, 'Magnitude')
    db_mag = torchaudio.functional.amplitude_to_DB(mag.abs(), 20, 1e-05, 1)[:,None,:,:] #[Batch_size,1, 128, 126]
    
    # Phase    
    phase = stft(audio, 'Phase')[:,None,:,:]

    #Mel spectograms
    mel_spec = mel_transform(audio)                   
    mel_spectogram = transforms.AmplitudeToDB()(mel_spec)
    
    # Stack them Magnitude+Phase+Melspectogram
    
    new_tensor = torch.Tensor(size=[audio.shape[0],3, 128, 126])
    for i in range(audio.shape[0]):
        new_tensor[i] = torch.cat((db_mag[i], phase[i], mel_spectogram[i]), dim=0)
    
    return new_tensor


def getData(batch_size = 32, num_workers = 0, pin_memory = False, percentage = 1):

    train_set = SubsetSC("training", percentage=percentage, batch_size=32)
    test_set = SubsetSC("testing", percentage=percentage)
    val_set = SubsetSC("validation", percentage=percentage)
    
    # creating Dataloaders
    train_loader = torch.utils.data.DataLoader( train_set, batch_size=batch_size,shuffle=True,drop_last=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory = pin_memory)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=False,collate_fn=collate_fn,num_workers=0)
    
    return train_loader, test_loader, val_loader



