import argparse
import json
import multiprocessing as mp
import os
import socket
from typing import List, Optional
from tqdm import tqdm
import random

import transformers
import torch
import torchaudio
import torch.distributed as dist
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import pipeline

from univoice.univoice_model import UniVoice
from univoice.constants import *

from univoice.utils import MelSpec, MelSpec_bigvGAN
from univoice.tensor_util import spec_to_figure, spec_to_figure_single

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--cfg_scale", type=float, required=True)
    
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    setup_seed(42) # random seed default=42

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, add_bos_token=True, add_eos_token=True)
    
    # load model
    model_config = AutoConfig.from_pretrained(args.ckpt_path)
    model_config.learn_sigma = True
    model_config.tokenizer_max_length = 1024
    model_config.tokenizer_padding_side = 'right'
    model_config.use_flash_attn = False
    # model_config.attn_implementation="flash_attention_2" if model_config.use_flash_attn==True else "eager"
    model_config.use_pos_embed = True
    model_config.decoder_t_embed = "add_before_speech_tokens"
    model_config.use_adaln_final_layer = True
    model_config.use_bi_attn_img_tokens = True   # or False for causal DiT
    model_config.add_pos_embed_each_layer = False
    model_config.use_hybrid_attn_mask = False
    model_config.audio_encoder_path = 'hf_ckpts/whisper-large-v3'
    model_config.speaker_encoder_path = 'hf_ckpts/wav2vec2-large-xlsr-53'
    model = UniVoice(
        model_config,
        llm_path = args.ckpt_path,
        tokenizer = tokenizer,
        cfg_scale = args.cfg_scale,
    )
    ckpt_type = args.ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(f"{args.ckpt_path}/pytorch_model.bin", device='cuda')
    else:
        checkpoint = torch.load(f"{args.ckpt_path}/pytorch_model.bin", map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    
    # wav_path for speaker
    wav_path = "data/LJ001-0001.wav"
    
    audio, source_sample_rate = torchaudio.load(wav_path)
    if audio.shape[0] > 1: # mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_sample_rate != 22050:   # whisper---16KHZ
        resampler = torchaudio.transforms.Resample(source_sample_rate, 22050)
        audio = resampler(audio)
    mel_spectrogram = MelSpec_bigvGAN(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        target_sample_rate=22050,
    )
    mel_spec = mel_spectrogram(audio)
    mel_spec = [mel_spec.squeeze(0).to('cuda')] # (D,T)
    speechs = [[]]
    flags = [[0]]


    duration = 6
    target_len = [int(duration*22050//256)]  # mel_spec[0].shape[1]. 
    text = ["At once the goat gave a leap, escaped from the soldiers and with bowed head rushed upon the Boolooroo".lower()]

    temp = torch.randn(1).to('cuda')
    with torch.inference_mode():
        mel_out, mel_gt = model.sample(
            input_ids=temp,
            attention_mask=temp,
            labels=temp,
            mel_spec=mel_spec,
            speechs=speechs,
            flags=flags,
            target_len=target_len,
            text=text,
            wav_path=[wav_path],
        )
    text_name = '_'.join(text[0].strip().split())
    os.makedirs('infers', exist_ok=True)
    spec_to_figure(mel_out, title="", file_name=f"infers/pred_{text_name}.png")
    # bigvagn vocoder
    from BigVGAN import bigvgan
    vocoder = bigvgan.BigVGAN.from_pretrained('hf_ckpts/bigvgan_22k', use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to('cuda')

    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = vocoder(mel_out.transpose(0,1).unsqueeze(0)) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
    wav_gen_float = wav_gen.squeeze(0).cpu()
    # wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype('int16') # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype

    torchaudio.save(f'infers/{text_name}.wav', wav_gen_float, 22050)


if __name__ == "__main__":
    main()
