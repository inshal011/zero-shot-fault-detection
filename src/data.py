"""Dataset utilities for MIMII and ToyADMOS.

Expected layouts (after extracting the Zenodo archives):

MIMII (https://zenodo.org/record/3384388), e.g. 6 dB SNR:
    <root>/fan/id_00/normal/00000000.wav
    <root>/fan/id_00/abnormal/00000000.wav
    ... same for pump / valve / slider, ids 00,02,04,06

ToyADMOS (https://zenodo.org/record/3351307), e.g. ToyCar:
    <root>/ToyCar/case1/NormalSound_IND/*.wav
    <root>/ToyCar/case1/AnomalousSound_IND/*.wav

The scanners below match on path keywords ('normal'/'abnormal'/'anomal')
so minor layout differences between archive versions still resolve.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

MIMII_MACHINES = ["fan", "pump", "slider", "valve"]
SR = 16000


def list_wavs(root: Path, machine: str):
    """Return (normal_files, abnormal_files) for one machine type."""
    normal, abnormal = [], []
    for p in sorted(root.rglob("*.wav")):
        parts_lower = [s.lower() for s in p.parts]
        if not any(machine.lower() in s for s in parts_lower):
            continue
        path_str = "/".join(parts_lower)
        if "abnormal" in path_str or "anomal" in path_str:
            abnormal.append(p)
        elif "normal" in path_str:
            normal.append(p)
    return normal, abnormal


class LogMel:
    def __init__(self, n_mels=128, n_fft=1024, hop=512, sr=SR):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __call__(self, wav):
        # wav: (channels, samples) -> mono log-mel (n_mels, frames), normalized
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        m = self.db(self.mel(wav)).squeeze(0)
        return (m - m.mean()) / (m.std() + 1e-6)


class PatchDataset(Dataset):
    """Cuts each recording's log-mel into fixed-width time patches.

    Items: (patch (1, n_mels, n_frames), domain_id, label, file_idx)
    label: 0 = normal, 1 = anomalous (labels only used at *evaluation*).
    """

    def __init__(self, files, domain_ids, labels, n_frames=64, n_mels=128,
                 patches_per_file=None, cache=True):
        self.files = [Path(f) for f in files]
        self.domain_ids = list(domain_ids)
        self.labels = list(labels)
        self.n_frames = n_frames
        self.fe = LogMel(n_mels=n_mels)
        self.patches_per_file = patches_per_file
        self.cache = {} if cache else None
        self.index = []  # (file_idx, start_frame)
        self._build_index()

    def _mel(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        wav, sr = torchaudio.load(str(self.files[i]))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        m = self.fe(wav)
        if self.cache is not None:
            self.cache[i] = m
        return m

    def _build_index(self):
        for i in range(len(self.files)):
            m = self._mel(i)
            n = m.shape[1]
            starts = list(range(0, max(1, n - self.n_frames + 1), self.n_frames // 2))
            if self.patches_per_file and len(starts) > self.patches_per_file:
                starts = random.sample(starts, self.patches_per_file)
            for s in starts:
                self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        m = self._mel(i)
        patch = m[:, s:s + self.n_frames]
        if patch.shape[1] < self.n_frames:  # pad short tails
            patch = torch.nn.functional.pad(patch, (0, self.n_frames - patch.shape[1]))
        return patch.unsqueeze(0), self.domain_ids[i], self.labels[i], i


def build_source_train(mimii_root, machines, n_frames=64, patches_per_file=8):
    """Training set: NORMAL clips only, from the source machine types."""
    files, doms, labels = [], [], []
    root = Path(mimii_root)
    for d, mch in enumerate(machines):
        normal, _ = list_wavs(root, mch)
        if not normal:
            raise FileNotFoundError(f"No normal wavs found for '{mch}' under {root}")
        files += normal
        doms += [d] * len(normal)
        labels += [0] * len(normal)
    return PatchDataset(files, doms, labels, n_frames=n_frames,
                        patches_per_file=patches_per_file)


def build_target_test(root, machine, n_frames=64, max_files_per_class=None):
    """Test set: normal + anomalous clips of the UNSEEN machine."""
    normal, abnormal = list_wavs(Path(root), machine)
    if max_files_per_class:
        normal = normal[:max_files_per_class]
        abnormal = abnormal[:max_files_per_class]
    if not normal or not abnormal:
        raise FileNotFoundError(
            f"Target machine '{machine}' under {root}: "
            f"{len(normal)} normal / {len(abnormal)} abnormal files found.")
    files = normal + abnormal
    labels = [0] * len(normal) + [1] * len(abnormal)
    doms = [-1] * len(files)
    return PatchDataset(files, doms, labels, n_frames=n_frames)
