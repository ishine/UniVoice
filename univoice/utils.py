import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torchaudio
from typing import Tuple, List
from librosa.filters import mel as librosa_mel_fn
import transformers
from univoice.constants import *
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import json
from importlib import import_module
import inspect

mel_basis_cache = {}
hann_window_cache = {}

# vocos
class MelSpec_hifigan(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=False,
            norm="slaney",
            onesided=True,
            mel_scale="slaney",
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(waveform.shape) == 2

        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel

class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        target_sample_rate=16000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(waveform.shape) == 2

        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel


mel_basis_cache = {}
hann_window_cache = {}
class MelSpec_bigvGAN(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        fmin=0,
        fmax=8000,
        center=False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, waveform):
        if self.dummy.device != waveform.device:
            self.to(waveform.device)

        device = waveform.device
        key = f"{self.n_fft}_{self.n_mel_channels}_{self.target_sample_rate}_{self.hop_length}_{self.win_length}_{self.fmin}_{self.fmax}_{device}"

        if key not in mel_basis_cache:
            mel = librosa_mel_fn(
                sr=self.target_sample_rate, 
                n_fft=self.n_fft, 
                n_mels=self.n_mel_channels, 
                fmin=self.fmin, 
                fmax=self.fmax)
            mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
            hann_window_cache[key] = torch.hann_window(self.win_length).to(device)

        mel_basis = mel_basis_cache[key]
        hann_window = hann_window_cache[key]

        padding = (self.n_fft - self.hop_length) // 2
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        
        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        # mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        return mel_spec


# copied from https://github.com/MonoFormer/MonoFormer
def preprocess_single_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], max_length=256, device='cuda'):
    """
    Steps to preprocess inputs:
    1. add special geenration tokens after inputs: <|im_start|><image><|im_end|>
    2. add bos token before inputs
    3. tokenize inputs
    4. replace tokens after bos and before <|im_start|> with padding tokens to form unconditional inputs
    5. concatenate conditional inputs and unconditional inputs along batch dimension
    6. create attention masks by masking padding tokens
    7. create noise speech indices
    """
    generation_tokens = f"{DEFAULT_SPEECH_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_SPEECH_END_TOKEN}"

    inputs = [f"{tokenizer.bos_token}{example}{generation_tokens}" for example in inputs]
    
    input_ids = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        # return_tensors="pt",
    )['input_ids']
    
    input_ids = [torch.tensor(i) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    # FIXME: replace pad token after <|im_start|> with <speech>, this is due to tokenizer cannot correctly tokenize <image> after <|im_start|>
    im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)
    im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    for cur_input_ids in input_ids:
        for idx in torch.where(cur_input_ids == im_start_token_id):
            if cur_input_ids[idx + 1] == tokenizer.pad_token_id:
                cur_input_ids[idx + 1] = speech_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    flags = [[0] for _ in range(len(input_ids))]
    speechs = [[] for _ in range(len(input_ids))]
 
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'flags': flags,
        'speechs': speechs,
    }

