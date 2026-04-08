import os
from typing import Tuple, Optional, Union
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor

# AudioMNIST: 60 speakers, speaker-based train/val/test split
TRAIN_SPEAKERS = [f"{i:02d}" for i in range(1, 49)]   # 01-48
VAL_SPEAKERS   = [f"{i:02d}" for i in range(49, 55)]  # 49-54
TEST_SPEAKERS  = [f"{i:02d}" for i in range(55, 61)]  # 55-60


def load_audiomnist_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    """Load one AudioMNIST sample.

    AudioMNIST filename format: ``{digit}_{speaker_id}_{utterance}.wav``
    inside a per-speaker directory, e.g. ``data/01/0_01_3.wav``.

    Returns:
        (waveform, sample_rate, label, speaker_id, utterance_number)
    """
    relpath = os.path.relpath(filepath, path)
    speaker_id = os.path.dirname(relpath)           # e.g. "01"
    filename   = os.path.basename(relpath)          # e.g. "0_01_3.wav"

    stem, _ = os.path.splitext(filename)            # e.g. "0_01_3"
    parts = stem.split("_")                         # ["0", "01", "3"]

    label            = parts[0]                     # digit string ("0" … "9")
    utterance_number = int(parts[2])

    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


class AudioMnistDataset(Dataset):
    """Dataset for AudioMNIST.

    AudioMNIST contains 30,000 audio samples of spoken digits (0-9)
    from 60 different speakers.

    Expected directory layout::

        {root}/
          data/
            01/
              0_01_0.wav
              ...
            02/
              ...
            60/
              ...

    Clone the dataset with::

        git clone https://github.com/soerenab/AudioMNIST.git

    Reference paper:
        https://www.sciencedirect.com/science/article/pii/S0016003223007536

    Args:
        root (str or Path): Path to the AudioMNIST root directory
            (the folder that contains the ``data/`` subdirectory).
        subset (str or None, optional):
            Select a subset [None, "training", "validation", "testing"].
            None means the whole dataset.

            Splits are speaker-based:
              - ``"training"``  : speakers 01–48
              - ``"validation"``: speakers 49–54
              - ``"testing"``   : speakers 55–60

            (Default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: Optional[str] = None,
    ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            "{'training', 'validation', 'testing'}."
        )

        root = os.fspath(root)
        self._path = os.path.join(root, "data")

        if not os.path.isdir(self._path):
            raise FileNotFoundError(
                f"AudioMNIST data directory not found at '{self._path}'. "
                "Clone the dataset with: git clone https://github.com/soerenab/AudioMNIST.git"
            )

        self.index_list = None
        self.subset = subset

        if subset == "training":
            allowed = set(TRAIN_SPEAKERS)
        elif subset == "validation":
            allowed = set(VAL_SPEAKERS)
        elif subset == "testing":
            allowed = set(TEST_SPEAKERS)
        else:
            allowed = None

        walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))

        if allowed is not None:
            self._walker = [
                w for w in walker
                if os.path.basename(os.path.dirname(w)) in allowed
            ]
        else:
            self._walker = walker

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): Index of the sample to load.

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        filepath = self._walker[n]

        waveform, sample_rate, label, speaker_id, utterance_number = load_audiomnist_item(
            filepath, self._path
        )

        # Mask the label for semi-supervised training
        if self.index_list is not None and self.subset == "training":
            if n in self.index_list:
                label = "None"

        return waveform, sample_rate, label, speaker_id, utterance_number

    def __len__(self) -> int:
        return len(self._walker)
