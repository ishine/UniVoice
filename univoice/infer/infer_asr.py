import argparse
import json
import multiprocessing as mp
import os
import socket
from typing import List, Optional

import transformers
import random
import numpy as np
import torch
import torchaudio
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import pipeline

from univoice.univoice_model import UniVoice
from univoice.constants import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_inputs(tokenizer: transformers.PreTrainedTokenizer, inputs: List[str], speechs: List[torch.Tensor], max_length=512, device='cuda'):
    """
    Currently, only support batch size 1.
    """
    assert len(inputs) == 1

    input_ids, attention_mask = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).values()
    
    if len(speechs) > 0:
        # FIXME: replace pad token after <|im_start|> with <speech>, this is due to tokenizer cannot correctly tokenize <speech> after <|im_start|>
        im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)
        im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)
        speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        for cur_input_ids in input_ids:
            for idx in torch.where(cur_input_ids == im_start_token_id):
                if cur_input_ids[idx + 1] == tokenizer.pad_token_id:
                    cur_input_ids[idx + 1] = speech_token_id

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        flags = [[1]]
    else:
        flags = []

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'speechs': [speechs],
        'flags': flags,
        't': torch.tensor([0]).to(device),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
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
        cfg_scale = 1,
    )
    ckpt_type = args.ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(f"{args.ckpt_path}/pytorch_model.bin", device='cuda')
    else:
        checkpoint = torch.load(f"{args.ckpt_path}/pytorch_model.bin", map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    

    feature_extracter = transformers.WhisperFeatureExtractor.from_pretrained('hf_ckpts/whisper-large-v3')

    # asr wav_path
    wav_path = "data/LJ001-0001.wav"
    audio, source_sample_rate = torchaudio.load(wav_path)
    if audio.shape[0] > 1: # mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_sample_rate != 16000:   # whisper---16KHZ
        resampler = torchaudio.transforms.Resample(source_sample_rate, 16000)
        audio = resampler(audio)
    
    mel_spec = feature_extracter(audio.numpy(), sampling_rate=16000).input_features[0]
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    # speechs and prompt
    speechs = [mel_spec.to('cuda')]
    prompt = f"{DEFAULT_SPEECH_START_TOKEN}{DEFAULT_PAD_TOKEN}{DEFAULT_SPEECH_END_TOKEN}\n"
    inputs = [f"{tokenizer.bos_token}{prompt}"]

    inputs_dict = preprocess_inputs(tokenizer, inputs, speechs)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs_dict['input_ids'],
            attention_mask=inputs_dict['attention_mask'],
            speechs=inputs_dict['speechs'],
            flags=inputs_dict['flags'],
            t=inputs_dict['t'],
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
        )
    output_ids = output_ids.replace("\n"," ").replace("<|im_end|>","")
    print(output_ids)


if __name__ == "__main__":
    main()