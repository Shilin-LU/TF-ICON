# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, name='image', timestamp=0, layernum=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]   
    
    pil_img = Image.fromarray(image_)
    display(pil_img)
    
    
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    
    # classifier-free guidance during inference
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond) 
    # noise_pred = guidance_scale * noise_prediction_text
    
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings]) 
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def register_attention_control(model, controller, center_row_rm, center_col_rm, target_height, target_width, width, height, top=None, left=None, bottom=None, right=None, inject_bg=False, segmentation_map=None, pseudo_cross=False): 
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None, encode=False, controller_for_inject=None, inject=False, layernum=None, main_height=None, main_width=None):
            torch.cuda.empty_cache()
            is_cross = context is not None
            h = self.heads

            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                        
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
                    
            # ref's location in ref image
            top_rr = int((0.5*(target_height - height))/target_height * main_height)  
            bottom_rr = int((0.5*(target_height + height))/target_height * main_height) 
            left_rr = int((0.5*(target_width - width))/target_width * main_width)  
            right_rr = int((0.5*(target_width + width))/target_width * main_width)  
            
            new_height = bottom_rr - top_rr
            new_width = right_rr - left_rr    
                
            step_height2, remainder = divmod(new_height, 2)
            step_height1 = step_height2 + remainder
            step_width2, remainder = divmod(new_width, 2)
            step_width1 = step_width2 + remainder      

            center_row = int(center_row_rm * main_height)
            center_col = int(center_col_rm * main_width)
            
            if pseudo_cross:
                               
                ref_init = rearrange(x[2], '(h w) c ->1 c h w', h=main_height).contiguous()
                context = ref_init[:, :, top_rr:bottom_rr, left_rr:right_rr]
                
                context = rearrange(context, '1 c h w ->1 (h w) c').contiguous()
                context = repeat(context, '1 ... -> b ...', b=2)
                
                if (sim.shape[1])**0.5 == 32:
                    seg_map = segmentation_map[::2, ::2].clone()
                elif (sim.shape[1])**0.5 == 16:
                    seg_map = segmentation_map[::4, ::4].clone()
                elif (sim.shape[1])**0.5 == 8:
                    seg_map = segmentation_map[::8, ::8].clone()
                else:
                    seg_map = segmentation_map.clone()
                    
                # record reference location
                ref_loc_masked = []
                for i in range(bottom_rr - top_rr):
                    for j in range(right_rr - left_rr):
                        if seg_map[top_rr:bottom_rr, left_rr:right_rr][i, j] == 1:
                            ref_loc_masked.append(int(i * (right_rr - left_rr) + j))        
                ref_loc_masked = torch.tensor((ref_loc_masked), device=x.device)

                if len(ref_loc_masked) == 0:
                    masked_context = context
                else:
                    masked_context = context[:, ref_loc_masked, :]
                             
                k = self.to_k(masked_context)
                v = self.to_v(masked_context)

                k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))
                sim = einsum('b i d, b j d -> b i j', q[int(q.shape[0]/2):], k) * self.scale
                
                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~mask, max_neg_value)
                
                if encode == False:
                    sim = controller(sim, is_cross, place_in_unet)

                sim = sim.softmax(dim=-1)
                out = einsum('b i j, b j d -> b i d', sim, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                
                out = torch.cat([out]*2, dim=0)
                
                del sim, k, v, masked_context, q, ref_loc_masked, context, ref_init, seg_map, mask
                
                return self.to_out(out)
            
            if encode == False:
                sim = controller(sim, is_cross, place_in_unet)
                
            if inject or inject_bg and is_cross == False:
                            
                if (sim.shape[1])**0.5 == 32:
                    seg_map = segmentation_map[::2, ::2]
                elif (sim.shape[1])**0.5 == 16:
                    seg_map = segmentation_map[::4, ::4]
                elif (sim.shape[1])**0.5 == 8:
                    seg_map = segmentation_map[::8, ::8]
                else:
                    seg_map = segmentation_map

                ref_loc = []
                ref_loc_masked = []
                for i in range(top_rr, bottom_rr):
                    for j in range(left_rr, right_rr):
                        ref_loc.append(int(i * (sim.shape[1])**0.5 + j))
                        if seg_map[i, j] == 1:
                            ref_loc_masked.append(int(i * (sim.shape[1])**0.5 + j))
                ref_loc_masked = torch.tensor((ref_loc_masked), device=x.device)
                
                ref_col_maksed1 = ref_loc_masked.repeat(len(ref_loc_masked)).unsqueeze(1)
                ref_col_maksed2 = ref_loc_masked.repeat_interleave(len(ref_loc_masked)).unsqueeze(1)
                
                # original location
                orig_loc_masked = []
                orig_loc = []
                
                orig_mask = torch.zeros_like(sim[h:])
                mask_for_realSA = torch.zeros_like(sim[h:])

                for i_seg, i in enumerate(range(center_row - step_height1, center_row + step_height2)):
                    for j_seg, j in enumerate(range(center_col - step_width1, center_col + step_width2)):
                        orig_loc.append(int(i * (sim.shape[1])**0.5 + j))
                        # within segmentation map
                        if seg_map[top_rr:bottom_rr, left_rr:right_rr][i_seg, j_seg] == 1:
                            orig_loc_masked.append(int(i * (sim.shape[1])**0.5 + j))
                orig_loc_masked = torch.tensor((orig_loc_masked), device=x.device)
                
                orig_col_masked1 = orig_loc_masked.repeat(len(orig_loc_masked)).unsqueeze(1)
                orig_col_masked2 = orig_loc_masked.repeat_interleave(len(orig_loc_masked)).unsqueeze(1)

                orig_loc = torch.tensor((orig_loc), device=x.device)
                                
                mask_for_realSA[:, orig_loc, :] = 1
                mask_for_realSA[:, :, orig_loc] = 1
                
                if place_in_unet == 'down':
                    
                    if inject_bg:
                        # inject background of the squared region
                        sim[h:] = controller_for_inject[0].attention_store['down_self'][layernum] * (1 - mask_for_realSA) + sim[h:] * mask_for_realSA

                    if inject and len(ref_col_maksed1) != 0:
                        # inject the pesudo cross attention
                        if len(orig_loc_masked) != 0:
                            sim[h:, :, orig_loc_masked] = controller_for_inject[2].attention_store['down_self'][layernum] # row injection
                            sim[h:, orig_loc_masked, :] = controller_for_inject[2].attention_store['down_self'][layernum].permute(0,2,1) # column injection
                        # inject the foreground in the squared region but masked by the segmentation map
                        sim[h:, orig_col_masked1, orig_col_masked2] = controller_for_inject[1].attention_store['down_self'][layernum][:, ref_col_maksed1, ref_col_maksed2] 
                    
                elif place_in_unet == 'up':  
                                        
                    if inject_bg:
                        # inject background of the squared region
                        sim[h:] = controller_for_inject[0].attention_store['up_self'][layernum] * (1 - mask_for_realSA) + sim[h:] * mask_for_realSA

                    if inject and len(ref_col_maksed1) != 0:
                        # inject the pesudo cross attention
                        if len(orig_loc_masked) != 0:
                            sim[h:, :, orig_loc_masked] = controller_for_inject[2].attention_store['up_self'][layernum] # row injection
                            sim[h:, orig_loc_masked, :] = controller_for_inject[2].attention_store['up_self'][layernum].permute(0,2,1) # column injection
                        # inject the foreground in the squared region but masked by the segmentation map
                        sim[h:, orig_col_masked1, orig_col_masked2] =  controller_for_inject[1].attention_store['up_self'][layernum][:, ref_col_maksed1, ref_col_maksed2] 
                    
                elif place_in_unet == 'mid':
                    
                    if inject_bg:
                        # inject background of the squared region
                        sim[h:] = controller_for_inject[0].attention_store['mid_self'][layernum] * (1 - mask_for_realSA) + sim[h:] * mask_for_realSA
        
                    if inject and len(ref_col_maksed1) != 0:   
                        # inject the pesudo cross attention
                        if len(orig_loc_masked) != 0:
                            sim[h:, :, orig_loc_masked] = controller_for_inject[2].attention_store['mid_self'][layernum] # row injection
                            sim[h:, orig_loc_masked, :] = controller_for_inject[2].attention_store['mid_self'][layernum].permute(0,2,1) # column injection
                        # inject the foreground in the squared region but masked by the segmentation map
                        sim[h:, orig_col_masked1, orig_col_masked2] = controller_for_inject[1].attention_store['mid_self'][layernum][:, ref_col_maksed1, ref_col_maksed2]

                del orig_mask, mask_for_realSA, orig_loc_masked, orig_col_masked1, orig_col_masked2, ref_col_maksed1, ref_col_maksed2

            sim = sim.softmax(dim=-1)
                
            out = einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            
            del sim, v, q, k, context
            
            return self.to_out(out)
            
        return forward

    def register_recr(net_, count, place_in_unet):
        if 'CrossAttention' in net_.__class__.__name__:
            if net_.to_k.in_features != 1024:
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1 
            else:
                return count
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    # sub_nets = model.unet.named_children()
    sub_nets = model.model.diffusion_model.named_children()
    for net in sub_nets:
        if "input" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
        
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words
