import torch
import os
import shutil
from tqdm import tqdm
import copy
import time
import glob
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from univoice.univoice import MonoFormerForCausalLM
from univoice.constants import *

from torch.nn import MSELoss, L1Loss
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from univoice.fm_scheduler import (
    FlowMatchingTrainer,
)
from univoice.utils import preprocess_single_inputs
from univoice.tensor_util import spec_to_figure
from peft import LoraConfig, TaskType, get_peft_model

def setup_tokenizer(model, tokenizer):
    """
    Add speech generation tokens to the tokenizer. And resize the embedding layer of the model to match the tokenizer vocab size.
    """
    vocab = tokenizer.get_vocab()
    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_tokens([DEFAULT_SPEECH_TOKEN, DEFAULT_SPEECH_START_TOKEN, DEFAULT_SPEECH_END_TOKEN], special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

class UniVoice(nn.Module):
    def __init__(
        self,
        config,
        llm_path,
        tokenizer,
        cfg_scale, 
    ):
        super(UniVoice, self).__init__()
        self.model = MonoFormerForCausalLM.from_pretrained(llm_path, config=config)
        self.tokenizer = tokenizer
        self.config = config

        self.cfg_scale = cfg_scale
        
        self.model.initialize_weights()
        self.model, self.tokenizer = setup_tokenizer(self.model, self.tokenizer)
        self.model.config.speech_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        
        # flow matching trainer
        self.trainer = FlowMatchingTrainer(self.model, sample_N=50)

       
    def forward(
        self, input_ids, attention_mask, labels, mel_spec, speechs, flags, target_len, text, wav_path
        # self, asr,tts, # for more efficient training
    ):

        outputs = {}
        loss = self.trainer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    speechs=speechs,  # understand
                    mel_spec=mel_spec, # generate
                    flags=flags,
                    target_len=target_len,
                    wav_path=wav_path,
                )

        outputs['loss'] = loss

        return outputs
    

    @torch.no_grad()
    def sample(
        self, input_ids, attention_mask, labels, mel_spec, speechs, flags, target_len, text, wav_path
    ):
        for i in range(1):
            
            z1 = torch.randn_like(mel_spec[i], device=input_ids.device)
            # y = input_ids[i]
            mel_gt = mel_spec[i]
            text = text[i]

            z = [z1,z1]
            y_null = [""] 
            y = [text] + y_null
            inputs_dict = preprocess_single_inputs(self.tokenizer, y)
            target_len = [target_len[i],target_len[i]]
            wav_path = [wav_path[i], wav_path[i]]

            cfg_scale = self.cfg_scale
            with torch.no_grad():
                mel_out,nfe = self.trainer.euler_sample(
                    input_ids=inputs_dict['input_ids'],
                    attention_mask=inputs_dict['attention_mask'],
                    labels=inputs_dict['input_ids'],
                    mel_spec=z,
                    speechs=speechs,
                    flags=inputs_dict['flags'],
                    target_len=target_len,
                    wav_path=wav_path,
                    guidance_scale=cfg_scale)
            mel_out = mel_out[:target_len[i],:]
        return mel_out, mel_gt.transpose(0,1)


    # text generation
    @torch.inference_mode()
    def generate(
        self, input_ids, attention_mask, speechs, flags, t, temperature, top_p, top_k, num_beams
    ):
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            flags=flags,
            speechs=speechs,
            t=t,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=256,
            use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        return outputs