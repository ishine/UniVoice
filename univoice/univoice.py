# modified form https://github.com/MonoFormer/MonoFormer
from transformers import LlamaForCausalLM
import math
from torch import nn
import functools
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import warnings
import random
import os
import numpy as np


import torch
import transformers
import torchaudio
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import WhisperModel, WhisperFeatureExtractor
from univoice.constants import IGNORE_INDEX
from univoice.modeling import MonoFormerModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

@dataclass
class MonoFormerCausalLMOutputWithPast(CausalLMOutputWithPast):
    x_out: Optional[torch.FloatTensor] = None
    x_indices: Optional[torch.LongTensor] = None


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear( hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


def modulate(x, shift, scale, mask):
    return x * (1 + scale.unsqueeze(1) * mask.unsqueeze(2)) + shift.unsqueeze(1) * mask.unsqueeze(2)

class WhisperProjection(nn.Module):
    def __init__(self, input_embedding_size=1280, output_embedding_size=960):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(250)
        self.proj = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(input_embedding_size)

    def forward(self, whisper_output):
        pooled = self.pool(whisper_output.transpose(-2, -1))
        normalized = self.ln1(pooled.transpose(-2, -1))
        projected = self.proj(normalized)
        return projected

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, use_adaln=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, 80, bias=True)
        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

    def forward(self, x, c_t, c_spk,spk_idx, mask):
        if self.use_adaln:
            x_indices = [i for i in range(len(spk_idx)) if spk_idx[i] != []]
            if x_indices==[]:
                x = None
                return x, x_indices
            x = [x[i] for i in x_indices] # x
            mask = [mask[i] for i in x_indices] # mask
            x = torch.stack(x,dim=0)
            mask = torch.stack(mask,dim=0)
            
            if c_spk is None:
                shift, scale = self.adaLN_modulation(c_t).chunk(2, dim=1)
                x = modulate(self.norm_final(x), shift, scale, mask)
            elif c_t.shape[0]>c_spk.shape[0]:
                c_t_new = [c_t[i] for i in x_indices]
                c_t = torch.stack(c_t_new, dim=0)
                c = c_t + c_spk
                # c = c_t
                shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
                x = modulate(self.norm_final(x), shift, scale, mask)

        x = self.linear(x)
        return x, x_indices

class MonoFormerForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = MonoFormerModel(config) 
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        
        self.x_embedder = nn.Linear(
            in_features = 80,
            out_features=config.hidden_size,
            bias=True,
        )

        self.t_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        )
        
        # audio understanding
        self.audio_encoder = None
        self.initialize_audio_modules(self.config.audio_encoder_path)

        # speaker_encoder init
        self.initialize_speaker_modules(self.config.speaker_encoder_path)

        self.final_layer = FinalLayer(config.hidden_size, use_adaln=self.config.use_adaln_final_layer)
        

    def initialize_weights(self):
        """
        Call this function to initialize the additional modules for DiT generation after loading pretrained weights.
        """
        for m in self.final_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.t_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.x_embedder.weight, 0)
        nn.init.constant_(self.x_embedder.bias, 0)

        if hasattr(self, 't_projector'):
            for m in self.t_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'adaLN_module'):
            for m in self.adaLN_module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
    
    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def embed_mel(self, x: List[torch.Tensor], target_len: List[torch.Tensor],  embedder: nn.Module) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        x_embed = []
        mel_size = []
        for i,mel in enumerate(x):
            H, W = mel.size()[1], target_len[i]
            mel_size.append((H, W))
            mel = embedder(mel)
            x_embed.append(mel)
        if len(x_embed) != 0:
            x_embed = pad_sequence(x_embed,batch_first=True,padding_value=0)
        return x_embed, mel_size

    def initialize_audio_modules(self, pretrained_path):
        self.config.audio_encoder_path = pretrained_path
        self.audio_encoder = WhisperModel.from_pretrained(pretrained_path).get_encoder()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_embedder = WhisperProjection(input_embedding_size=1280, output_embedding_size=960)
    
    def initialize_speaker_modules(self, pretrained_path):
        self.config.speaker_encoder_path = pretrained_path
        self.spk_encoder = transformers.Wav2Vec2ForPreTraining.from_pretrained(pretrained_path)
        for param in self.spk_encoder.parameters():
            param.requires_grad = False
        self.speaker_embedder = nn.Linear(
            in_features = 1024,
            out_features=960,
            bias=True,
        )
        
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, x_t, flags, target_len,wav_path, **kwargs
    ):
        # print('input_ids:',input_ids.shape) # (2, T) for tts, (1,T) for asr
        if input_ids.shape[1] == 1:
            wav_paths = wav_path
            spk_embs = []
            spk_idx = []
            speaker_embeds=None
          
            model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                'inputs_embeds': None,
                "labels": labels,
                'speech_token_spans': None,
                'speech_sizes': None,
                'c_embeds': None,
                't_embeds': t,
                'c_embeds_mask': None,
                'speaker_embeds': speaker_embeds,  # for multi-speaker
                'spk_idx': spk_idx,
            }
            return model_inputs

        if speechs is None:
            speechs = []
        
        speech_inputs = []
        x_t_inputs = []
        # generate
        for bid in range(len(flags)):
            for i in range(len(flags[bid])):
                if flags[bid][i] == 0:
                    x_t_inputs.append(x_t[bid])
        # understanding
        for speech_list in speechs:
            for speech in speech_list:
                speech_inputs.append(speech)

        x_t_embeds, x_t_sizes = self.embed_mel(x_t_inputs, target_len, self.x_embedder) 
        # asr audio encoder
        if getattr(self, 'audio_embedder', None) is not None and len(speech_inputs) > 0:

            speech_features = self.audio_encoder(torch.stack(speech_inputs)).last_hidden_state
            speech_features = self.audio_embedder(speech_features)
        else:
            speech_features = []


        speech_embeds = []
        speech_sizes = []
        speech_gen_idx = 0
        speech_und_idx = 0

        for bid in range(len(flags)):
            speech_size = None
            for i in range(len(flags[bid])):
                if flags[bid][i] == 0:
                    speech_embeds.append(x_t_embeds[speech_gen_idx])
                    speech_size = x_t_sizes[speech_gen_idx]
                    speech_gen_idx += 1
                elif flags[bid][i] == 1:
                    speech_embeds.append(speech_features[speech_und_idx])
                    speech_und_idx += 1
            speech_sizes.append(speech_size)


        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        
        speech_token_spans = []
        cur_speech_idx = 0
        
        t_embeds = self.t_embedder(t)  # (batch_size, hidden_size)
        # process wav_paths
        wav_paths = wav_path
        spk_embs = []
        spk_idx = []
        speaker_embeds=None
        if speech_gen_idx != 0:
            i = 0
            for wav_path in wav_paths:
                if wav_path != '':
                    spk_idx.append([i])
                    wave, sr = torchaudio.load(wav_path)
                    # mono
                    if wave.shape[0] > 1:
                        wave = torch.mean(wave, dim=0, keepdim=True)
                    # resample
                    if sr != 16000:   # whisper---16KHZ
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        wave = resampler(wave)

                    outputs = self.spk_encoder(wave.to(t.device), output_hidden_states=True)
                    spk_emb =  outputs.hidden_states[1].mean(1) 
                    spk_embs.append(spk_emb)
                    i+=1
                else:
                    spk_idx.append([])
            spk_embs = torch.cat(spk_embs, dim=0) # (B,D)
            speaker_embeds = self.speaker_embedder(spk_embs)

        if self.config.decoder_t_embed == 'add_before_speech_tokens':
            t_tokens = self.t_projector(t_embeds)  # (batch_size, hidden_size)

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speechs = (cur_input_ids == self.config.speech_token_index).sum()
            if num_speechs == 0 or len(flags[batch_idx]) == 0:
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                speech_token_spans.append([])
                continue
            
            speech_token_indices = [-1] + torch.where(cur_input_ids == self.config.speech_token_index)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_speech_token_spans = []

            for i in range(num_speechs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_speechs:
                    if self.config.decoder_t_embed == 'add_before_speech_tokens':
                        cur_new_input_embeds.append(t_tokens[batch_idx:batch_idx+1]) # add t condition
                        if spk_embs != [] and spk_idx[batch_idx] != []:
                            cur_new_input_embeds.append(speaker_embeds[spk_idx[batch_idx][0]].unsqueeze(0)) # add spk condition
                            cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_speech_features = speech_embeds[cur_speech_idx]
                    cur_speech_idx += 1 
                    speech_token_start_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_input_embeds.append(cur_speech_features)
                    speech_token_end_idx = torch.cat(cur_new_input_embeds).shape[0]
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    if flags[batch_idx][i] == 0:
                        cur_speech_token_spans = (speech_token_start_idx, speech_token_end_idx)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            speech_token_spans.append(cur_speech_token_spans)

        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        #===================================
        #  generate condition embedding mask
        #===================================
        c_embeds_mask = torch.zeros(new_input_embeds.shape[:2]).to(t_embeds.device)  # (batch_size, seq_len)
        for i, span in enumerate(speech_token_spans):
            if span:
                # c_embeds_mask[i, span[0]:span[1]] = 1    
                c_embeds_mask[i, span[0]:span[0]+target_len[i]] = 1
        
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        model_inputs = {
            'input_ids': None,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': new_input_embeds,
            'labels': new_labels,
            'speech_token_spans': speech_token_spans,
            'speech_sizes': speech_sizes,
            'speaker_embeds': speaker_embeds,  # for multi-speaker
            'spk_idx': spk_idx,
            't_embeds': t_embeds,
            'c_embeds_mask': c_embeds_mask,
        }

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speechs: Optional[List[torch.FloatTensor]] = None,
        t: Optional[torch.FloatTensor] = None,
        x_t: Optional[torch.FloatTensor] = None,
        target_len: Optional[torch.Tensor] = None,
        flags: Optional[List[torch.LongTensor]] = None,
        wav_path:Optional[str] = None, # wav_path
        # args for generate
        speech_token_spans: Optional[List[Tuple[int, int]]] = None,
        speech_sizes: Optional[List[Tuple[int, int]]] = None,
        t_embeds: Optional[torch.FloatTensor] = None,
        c_embeds_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        spk_idx: Optional[List[torch.LongTensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # for inference 
        if inputs_embeds is None:
            model_inputs = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, speechs, t, x_t, flags, target_len, wav_path)
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, speech_token_spans, speech_sizes, t_embeds, c_embeds_mask, speaker_embeds, spk_idx = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['speech_token_spans'],
                model_inputs['speech_sizes'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds_mask'],
                model_inputs['speaker_embeds'], # for multi-speaker
                model_inputs['spk_idx'],
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            c_embeds_mask=c_embeds_mask,
        )

        hidden_states = outputs['last_hidden_state'] 

        '''
        Compute outputs for generation losses
        '''
        x_out = []
        B = len(hidden_states)
        x_indices = []

        if c_embeds_mask is not None:
            x, x_indices = self.final_layer(hidden_states, t_embeds, speaker_embeds,spk_idx, c_embeds_mask)
            speech_token_spans = [speech_token_spans[i] for i in range(len(speech_token_spans)) if speech_token_spans[i] != []]
            if x == None:
                x_out=[]
            else:
                for i in range(x.shape[0]):
                    span = speech_token_spans[i]
                    if span:
                        speech_tokens = x[i, span[0]:span[1], :]
                        x_out.append(speech_tokens)

        x_indices = torch.tensor(x_indices)


        ''' 
        Compute the language modeling loss
        '''
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # only compute loss for samples not used for diffusion generation
            lm_indices = []
            lm_indices = torch.tensor([i for i in range(len(hidden_states)) if i not in x_indices], dtype=torch.long)
            if len(lm_indices) == 0:
                loss = torch.tensor(0., device=logits.device)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[lm_indices, :-1, :].contiguous()
                shift_labels = labels[lm_indices, 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        
        return MonoFormerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            x_indices=x_indices,
            x_out=x_out,
        )
        


    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        t = kwargs.pop("t", None)
        speechs = kwargs.pop("speechs", None)
        x_t = kwargs.pop("x_t", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        flags = kwargs.pop("flags", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        is_multimodal = False
        for i in range(len(flags)):
            if len(flags[i]) > 0:
                is_multimodal = True
                break
        
        if is_multimodal:
            model_inputs = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                speechs,
                t,
                x_t,
                flags,
                None, 
                None
            )
            _, position_ids, attention_mask, _, inputs_embeds, _, speech_token_spans, speech_sizes, t_embeds, c_embeds_mask, speaker_embeds, spk_idx  = (
                model_inputs['input_ids'], 
                model_inputs['position_ids'], 
                model_inputs['attention_mask'], 
                model_inputs['past_key_values'], 
                model_inputs['inputs_embeds'], 
                model_inputs['labels'],
                model_inputs['speech_token_spans'],
                model_inputs['speech_sizes'],
                model_inputs['t_embeds'],
                model_inputs['c_embeds_mask'],
                model_inputs['speaker_embeds'], # for multi-speaker
                model_inputs['spk_idx'],
            )
            kwargs.update({
                'speech_token_spans': speech_token_spans,
                'speech_sizes': speech_sizes,
                't_embeds': t_embeds,
                'c_embeds_mask': c_embeds_mask,
                'speaker_embeds': speaker_embeds,
                'spk_idx': spk_idx,
                'flags': flags,
                'speechs': speechs
            })
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        attention_mask = kwargs.pop("attention_mask", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        inputs.update(kwargs)
        return inputs
