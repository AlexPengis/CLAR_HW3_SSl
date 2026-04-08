#data /home/alexey/DLS/DLS_Spring_2026_Speach/HW_SSL3_DLS_speach2026/AudioMNIST
#%% md
## AudioSet

#Dataset with 10 numbers pronounced
```
!git clone https://github.com/soerenab/AudioMNIST.git
```


#!to do :
import os
from typing import Tuple, Optional, Union
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor



def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output

def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number



class AudioMnistDataset(Dataset):
    """Create a Dataset for Speech Commands.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        root = "/home/alexey/DLS/DLS_Spring_2026_Speach/HW_SSL3_DLS_speach2026/AudioMNIST"
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://www.sciencedirect.com/science/article/pii/S0016003223007536>`_. (Default: ``None``)
    """

    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False,
                 subset: Optional[str] = None,
                 ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
                "When `subset` not None, it must take a value from "
                + "{'training', 'validation', 'testing'}."
        )

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        self.index_list = None
        self.subset = subset

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [
                w for w in walker
                if HASH_DIVIDER in w
                   and EXCEPT_FOLDER not in w
                   and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]

        waveform, sample_rate, label, speaker_id, utterance_number = load_speechcommands_item(fileid,
                                                                                              self._path)

        # If training and the index is in the index list to be masked, the label is update as None

        if self.index_list is not None:
            if self.subset == 'training':
                if n in self.index_list:
                    label = 'None'

        return waveform, sample_rate, label, speaker_id, utterance_number

    def __len__(self) -> int:
        return len(self._walker)