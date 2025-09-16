import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from generate import add_gumbel_noise, get_num_transfer_tokens, get_transfer_index, get_transfer_index_dynamic
import logging
logger = logging.getLogger(__name__)
def powers_of_two_in_range(min_block_length, max_block_length):
    result = []
    n = 1
    while True:
        power = 2 ** n
        if power > max_block_length:
            break
        if power >= min_block_length:
            result.append(power)
        n += 1
    return result

@ torch.no_grad()
def generate_with_dual_dynamic_block_length(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            min_block_length=4, max_block_length=64,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
            sub_block_ratio=0.5):
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
        sub_block_ratio: The ratio of sub-block to the block. The actual decoding length is block_length * sub_block_ratio.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    assert gen_length % min_block_length == 0 and gen_length % max_block_length == 0, "gen_length must be divisible by min_block_length and max_block_length"
    valid_block_lengths = powers_of_two_in_range(min_block_length, max_block_length)
    steps = 1
    assert sub_block_ratio > 0 and sub_block_ratio <= 0.5, "sub_block_ratio must be between 0 and 0.5"
    assert int(block_length * sub_block_ratio)> 0 and int(block_length * sub_block_ratio) % 2 == 0, "block_length * sub_block_ratio must be even and positive"

    nfe = 0  
    current_block_start = prompt.shape[1]
    
    while current_block_start < x.shape[1]:
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        # run single model forward with prefix cache with the largest block length, to determine the block length
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l     
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        for block_length in valid_block_lengths:
            avg_confidence = x0_p[:, current_block_start:current_block_start+block_length].mean(dim=-1)
            current_block_end = current_block_start + block_length
            if current_block_end >= x.shape[1]:
                break
            if avg_confidence.mean() < 0.2:
                break

        logger.info(f"Block length: {block_length}, avg confidence: {avg_confidence.mean()}")

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for k in range(int(1/sub_block_ratio)):
            sub_block_start = current_block_start + k * int(block_length * sub_block_ratio)
            sub_block_end = min(current_block_start + (k + 1) * int(block_length * sub_block_ratio), current_block_end)

            mask_index = (x == mask_id)
            mask_index[:, sub_block_end:] = 0
            mask_index[:, :sub_block_start] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                current_block_start = current_block_end
                break
            nfe += 1
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            for k in range(int(1/sub_block_ratio)):
                sub_block_start = current_block_start + k * int(block_length * sub_block_ratio)
                sub_block_end = min(current_block_start + (k + 1) * int(block_length * sub_block_ratio), current_block_end)

                mask_index = (x == mask_id)
                mask_index[:, sub_block_end:] = 0
                mask_index[:, :sub_block_start] = 0
                slice_start = sub_block_start - current_block_start
                slice_end = sub_block_end - current_block_start
                sub_block_logits = logits[:, slice_start:slice_end]


                if factor is None:
                    x0, transfer_index = get_transfer_index(sub_block_logits, temperature, remasking, mask_index, 
                                                    x[:, sub_block_start:sub_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                else:
                    x0, transfer_index = get_transfer_index_dynamic(sub_block_logits, temperature, remasking, mask_index, 
                                                    x[:, sub_block_start:sub_block_end], None, factor)
                x[:, sub_block_start:sub_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


@torch.no_grad() 
def generate_with_prefix_dynamic_block_length(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    min_block_length: int = 4,
    max_block_length: int = 64,
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    sub_block_ratio=0.5
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
        sub_block_ratio: The ratio of sub-block to the block. The actual decoding length is block_length * sub_block_ratio.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % min_block_length == 0 and gen_length % max_block_length == 0, "gen_length must be divisible by min_block_length and max_block_length"
    valid_block_lengths = powers_of_two_in_range(min_block_length, max_block_length)
    steps = 1
    nfe = 0
    assert sub_block_ratio > 0 and sub_block_ratio <= 0.5, "sub_block_ratio must be between 0 and 0.5"
    assert int(block_length * sub_block_ratio)> 0 and int(block_length * sub_block_ratio) % 2 == 0, "block_length * sub_block_ratio must be even and positive"

    current_block_start = prompt.shape[1]
    
    while current_block_start < x.shape[1]:
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        # run single model forward to determine the block length
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l     
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        for block_length in valid_block_lengths:
            avg_confidence = x0_p[:, current_block_start:current_block_start+block_length].mean(dim=-1)
            current_block_end = current_block_start + block_length
            if current_block_end >= x.shape[1]:
                break
            if avg_confidence.mean() < 0.2:
                break

        logger.info(f"Block length: {block_length}, avg confidence: {avg_confidence.mean()}")

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for k in range(int(1/sub_block_ratio)):
            sub_block_start = current_block_start + k * int(block_length * sub_block_ratio)
            sub_block_end = min(current_block_start + (k + 1) * int(block_length * sub_block_ratio), current_block_end)

            mask_index = (x == mask_id)
            mask_index[:, sub_block_end:] = 0
            mask_index[:, :sub_block_start] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]


        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values


        nfe += 1
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                current_block_start = current_block_end
                break
            nfe += 1

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            for k in range(int(1/sub_block_ratio)):
                sub_block_start = current_block_start + k * int(block_length * sub_block_ratio)
                sub_block_end = min(current_block_start + (k + 1) * int(block_length * sub_block_ratio), current_block_end)

                mask_index = (x == mask_id)
                mask_index[:, sub_block_end:] = 0
                mask_index[:, :sub_block_start] = 0
                slice_start = sub_block_start - current_block_start
                slice_end = sub_block_end - current_block_start
                sub_block_logits = logits[:, slice_start:slice_end]


                if factor is None:
                    x0, transfer_index = get_transfer_index(sub_block_logits, temperature, remasking, mask_index, 
                                                    x[:, sub_block_start:sub_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                else:
                    x0, transfer_index = get_transfer_index_dynamic(sub_block_logits, temperature, remasking, mask_index, 
                                                    x[:, sub_block_start:sub_block_end], None, factor)
                x[:, sub_block_start:sub_block_end][transfer_index] = x0[transfer_index]
            i += 1


    return x, nfe
