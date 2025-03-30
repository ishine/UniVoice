from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class FlowMatchingTrainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        init_type="gaussian",
        noise_scale=1.0,
        reflow_t_schedule="uniform",
        use_ode_sampler="euler",
        ode_tol=1e-5,
        sample_N=32,
    ):
        super().__init__()
        self.model = model
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.reflow_t_schedule = reflow_t_schedule
        self.use_ode_sampler = use_ode_sampler
        self.ode_tol = ode_tol
        self.sample_N = sample_N
        self.T = 1
        self.eps = 1e-3
        print("Init. Distribution Variance:", self.noise_scale)
        print("ODE Tolerence:", self.ode_tol)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        speechs,
        mel_spec,
        flags,
        target_len,
        wav_path,
    ):

        mel_spec = pad_sequence(
            [mel.transpose(0,1) for mel in mel_spec], batch_first=True, padding_value=0
        )
        
        x_0 = mel_spec
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - self.eps) + self.eps
        t_expand = t.view(-1, 1, 1).repeat(
            1, x_0.shape[1], x_0.shape[2]
        )
        noise_tensor = torch.randn_like(mel_spec)
        target = x_0 - noise_tensor
        perturbed_data = t_expand * x_0 + (1 - t_expand) * noise_tensor
        x_t = [p for p in perturbed_data]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speechs=speechs,
            flags=flags,
            t=t * 999,
            x_t=x_t,
            target_len=target_len,
            wav_path=wav_path,
        )
        
        model_output = outputs['x_out']
        lm_loss = outputs['loss']
        x_indices = outputs['x_indices']
        
        if model_output==[]:
            return lm_loss*0.01
        else:
            x_out = torch.stack(model_output,dim=0)
            target = torch.stack([target[i] for i in x_indices],dim=0)
            fm_loss = F.mse_loss(x_out, target, reduction="none").mean([1, 2]).mean()
            return fm_loss+lm_loss*0.01

    @torch.no_grad()
    def euler_sample(
        self, 
        input_ids,
        attention_mask,
        labels,
        speechs,
        mel_spec,
        flags,
        target_len,
        wav_path,
        guidance_scale
    ):
        device = self.model.device
        x = [torch.randn_like(mel.transpose(0,1)) for mel in mel_spec]
        # uniform
        dt = 1.0 / self.sample_N
        eps = 1e-3
        for i in range(self.sample_N):
            num_t = i / self.sample_N * (self.T - eps) + eps
            t = torch.ones(len(mel_spec), device=device) * num_t

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                speechs=speechs,
                flags=flags,
                t=t * 999,
                x_t=x,
                target_len=target_len,
                wav_path=wav_path,
            )
            # perform guidance
            model_output = outputs['x_out']
            
            noise_pred_text, noise_pred_uncond = model_output
            pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )


            pred_sigma = pred 
            
            x = (
                x[0].detach().clone()
                + pred_sigma * dt
            )
            x = [x,x]


        nfe = self.sample_N
        return x[0], nfe
