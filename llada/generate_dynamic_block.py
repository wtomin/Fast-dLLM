import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .generate import add_gumbel_noise, get_num_transfer_tokens, get_transfer_index, get_transfer_index_dynamic

@torch.no_grad() 
def generate_with_dynamic_block_length(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    min_block_length: int = 16,
    max_block_length: int = 64,
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None
) -> tuple[torch.Tensor, int]:
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        min_block_length: Minimal block length, less than gen_length.
        max_block_length: Maximal block length, less than gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % min_block_length == 0 and gen_length % max_block_length == 0, "gen_length must be divisible by min_block_length and max_block_length"

    output = model(x, use_cache=True)
    # get prefill kv cache for each block
    past_key_values = output.past_key_values
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            new_past_key_values[i] += (past_key_values[i][j][:, :, :prompt.shape[1]],)
    
    last_past_key_values = new_past_key_values

    steps = 1
    nfe = 0

    # get the first block length
    current_block_start = prompt.shape[1]
    
    while current_block_start < x.shape[1]:
        # run single model forward with cache, determine the block length
        output = model(x[:, current_block_start:], past_key_values=last_past_key_values, use_cache=True)
        logits = output.logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l     
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        for block_length in range(min_block_length, max_block_length + 1):
            avg_confidence = x0_p[:, :block_length].mean(dim=-1)
            current_block_end = current_block_start + block_length
            if current_block_end >= gen_length:
                break
            if avg_confidence.mean() < 0.2:
                break
        nfe += 1
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                output = model(x, use_cache=True)
                # get prefill kv cache for each block
                past_key_values = output.past_key_values
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :, :prompt.shape[1]],)
                
                last_past_key_values = new_past_key_values
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=last_past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1

            past_key_values = output.past_key_values


    return x, nfe
