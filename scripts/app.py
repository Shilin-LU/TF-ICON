import gradio as gr
import os
import torch
from omegaconf import OmegaConf
import PIL
import torch
import numpy as np
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


DICT_MASK = {
    'x_1': 0,
    'y_1': 0,
    'x_2': 0,
    'y_2': 0,
}
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
CONFIG_PATH = "./configs/stable-diffusion/v2-inference.yaml"
CKPT_PATH = "/path/to/sd2/ckpt/v2-1_512-ema-pruned.ckpt"
CONFIG = OmegaConf.load(CONFIG_PATH) 

    
def load_img(image, SCALE, pad=False, seg_map=False, target_size=None):
    if seg_map:
        # Load the input image and segmentation map
        # image = Image.open(path).convert("RGB")
        # seg_map = Image.open(seg).convert("1")

        seg_map = seg_map.convert("1")
        # Get the width and height of the original image
        w, h = image.size

        # Calculate the aspect ratio of the original image
        aspect_ratio = h / w

        # Determine the new dimensions for resizing the image while maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = int(SCALE * 256 / aspect_ratio)
            new_h = int(SCALE * 256)
        else:
            new_w = int(SCALE * 256)
            new_h = int(SCALE * 256 * aspect_ratio)

        # Resize the image and the segmentation map to the new dimensions
        image_resize = image.resize((new_w, new_h))
        segmentation_map_resize = cv2.resize(np.array(seg_map).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad the segmentation map to match the target size
        padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
        start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
        start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
        padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0], start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize

        # Create a new RGB image with the target size and place the resized image in the center
        padded_image = Image.new("RGB", target_size)
        start_x = (target_size[0] - image_resize.width) // 2
        start_y = (target_size[1] - image_resize.height) // 2
        padded_image.paste(image_resize, (start_x, start_y))

        # Update the variable "image" to contain the final padded image
        image = padded_image
    else:
        # image = Image.open(path).convert("RGB")
        w, h = image.size        
        # print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        w = h = 512
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    if pad or seg_map:
        return 2. * image - 1., new_w, new_h, padded_segmentation_map
    
    return 2. * image - 1., w, h 


def load_model_and_get_prompt_embedding(model, scale, device, prompts, inv=False):
           
    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
        c = uc = inv_emb
    else:
        inv_emb = None
        
    if scale != 1.0:
        uc = model.get_learned_conditioning([""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)
        
    return c, uc, inv_emb
    
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model

MODEL = load_model_from_config(CONFIG, CKPT_PATH, DEVICE)  
MODEL.to(DEVICE)

def tficon(init_img, ref_img, seg, prompt, dpm_order, dpm_steps, tau_a, tau_b, domain, seed, scale):

    n_samples = 1
    precision = "autocast"
    gpu = 'cuda'
    ddim_eta = 0.0
    dpm_order = int(dpm_order[0])
    
    # The scale used in the paper
    # if domain == 'Cross Domain':
    #     scale = 5.0
    # elif domain == 'Real Domain':
    #     scale = 2.5

    scale = scale
    
    device = DEVICE
    model = MODEL
    batch_size = n_samples
    sampler = DPMSolverSampler(model)
    
    seed_everything(seed)
    # img = cv2.imread(mask, 0)
    # # Threshold the image to create binary image
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # # Find the contours of the white region in the image
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Find the bounding rectangle of the largest contour
    # x, y, new_w, new_h = cv2.boundingRect(contours[0])
    # Calculate the center of the rectangle
    
    x = DICT_MASK['x_1']
    y = DICT_MASK['y_1']
    new_w = DICT_MASK['x_2'] - DICT_MASK['x_1']
    new_h = DICT_MASK['y_2'] - DICT_MASK['y_1']
    
    center_x = x + new_w / 2
    center_y = y + new_h / 2
    # Calculate the percentage from the top and left
    center_row_from_top = round(center_y / 512, 2)
    center_col_from_left = round(center_x / 512, 2)

    aspect_ratio = new_h / new_w
    
    if aspect_ratio > 1:  
        mask_scale = new_w * aspect_ratio / 256  
        mask_scale = new_h / 256
    else:  
        mask_scale = new_w / 256
        mask_scale = new_h / (aspect_ratio * 256) 
            
    # mask_scale = round(mask_scale, 2)
    
    # =============================================================================================

    data = [batch_size * [prompt]]
    # read background image              
    init_image, target_width, target_height = load_img(init_img, mask_scale)
    init_image = repeat(init_image.to(device), '1 ... -> b ...', b=batch_size)
    save_image = init_image.clone()

    # read foreground image and its segmentation map
    ref_image, width, height, segmentation_map  = load_img(ref_img, mask_scale, seg_map=seg, target_size=(target_width, target_height))
    ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=batch_size)

    segmentation_map_orig = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
    segmentation_map_save = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 3 ...', b=batch_size)
    segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(device)

    top_rr = int((0.5*(target_height - height))/target_height * init_image.shape[2])  # xx% from the top
    bottom_rr = int((0.5*(target_height + height))/target_height * init_image.shape[2])  
    left_rr = int((0.5*(target_width - width))/target_width * init_image.shape[3])  # xx% from the left
    right_rr = int((0.5*(target_width + width))/target_width * init_image.shape[3]) 

    center_row_rm = int(center_row_from_top * target_height)
    center_col_rm = int(center_col_from_left * target_width)

    step_height2, remainder = divmod(height, 2)
    step_height1 = step_height2 + remainder
    step_width2, remainder = divmod(width, 2)
    step_width1 = step_width2 + remainder
        
    # compositing in pixel space for same-domain composition
    save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] \
            = save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2].clone() \
            * (1 - segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]) \
            + ref_image[:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
            * segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]

    # save the mask and the pixel space composited image
    save_mask = torch.zeros_like(init_image) 
    save_mask[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] = 1

    # image = Image.fromarray(((save_image/torch.max(save_image.max(), abs(save_image.min())) + 1) * 127.5)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
    precision_scope = autocast if precision == "autocast" else nullcontext
    
    # image composition
    with torch.no_grad():
        with precision_scope("cuda"):
            for prompts in data:
                print(prompts)
                c, uc, inv_emb = load_model_and_get_prompt_embedding(model, scale, device, prompts, inv=True)
                
                if domain == 'Real Domain': # same domain
                    init_image = save_image
                
                T1 = time.time()
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  
                
                # ref's location in ref image in the latent space
                top_rr = int((0.5*(target_height - height))/target_height * init_latent.shape[2])  
                bottom_rr = int((0.5*(target_height + height))/target_height * init_latent.shape[2])  
                left_rr = int((0.5*(target_width - width))/target_width * init_latent.shape[3])  
                right_rr = int((0.5*(target_width + width))/target_width * init_latent.shape[3]) 
                                        
                new_height = bottom_rr - top_rr
                new_width = right_rr - left_rr
                
                step_height2, remainder = divmod(new_height, 2)
                step_height1 = step_height2 + remainder
                step_width2, remainder = divmod(new_width, 2)
                step_width1 = step_width2 + remainder
                
                center_row_rm = int(center_row_from_top * init_latent.shape[2])
                center_col_rm = int(center_col_from_left * init_latent.shape[3])
                
                param = [max(0, int(center_row_rm - step_height1)), 
                        min(init_latent.shape[2] - 1, int(center_row_rm + step_height2)),
                        max(0, int(center_col_rm - step_width1)), 
                        min(init_latent.shape[3] - 1, int(center_col_rm + step_width2))]
                
                ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
            
                shape = [init_latent.shape[1], init_latent.shape[2], init_latent.shape[3]]
                z_enc, _ = sampler.sample(steps=dpm_steps,
                                        inv_emb=inv_emb,
                                        unconditional_conditioning=uc,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        eta=ddim_eta,
                                        order=dpm_order,
                                        x_T=init_latent,
                                        width=width,
                                        height=height,
                                        DPMencode=True,
                                        )
                
                z_ref_enc, _ = sampler.sample(steps=dpm_steps,
                                            inv_emb=inv_emb,
                                            unconditional_conditioning=uc,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            eta=ddim_eta,
                                            order=dpm_order,
                                            x_T=ref_latent,
                                            DPMencode=True,
                                            width=width,
                                            height=height,
                                            ref=True,
                                            )
                
                samples_orig = z_enc.clone()

                # inpainting in XOR region of M_seg and M_mask
                z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                    = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                    * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                    + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) \
                    * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])

                samples_for_cross = samples_orig.clone()
                samples_ref = z_ref_enc.clone()
                samples = z_enc.clone()

                # noise composition
                if domain == 'Cross Domain': 
                    samples[:, :, param[0]:param[1], param[2]:param[3]] = torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) 
                    # apply the segmentation mask on the noise
                    samples[:, :, param[0]:param[1], param[2]:param[3]] \
                            = samples[:, :, param[0]:param[1], param[2]:param[3]].clone() \
                            * (1 - segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]) \
                            + z_ref_enc[:, :, top_rr: bottom_rr, left_rr: right_rr].clone() \
                            * segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]
                
                mask = torch.zeros_like(z_enc, device=device)
                mask[:, :, param[0]:param[1], param[2]:param[3]] = 1
                                    
                samples, _ = sampler.sample(steps=dpm_steps,
                                            inv_emb=inv_emb,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            order=dpm_order,
                                            x_T=[samples_orig, samples.clone(), samples_for_cross, samples_ref, samples, init_latent],
                                            width=width,
                                            height=height,
                                            segmentation_map=segmentation_map,
                                            param=param,
                                            mask=mask,
                                            target_height=target_height, 
                                            target_width=target_width,
                                            center_row_rm=center_row_from_top,
                                            center_col_rm=center_col_from_left,
                                            tau_a=tau_a,
                                            tau_b=tau_b,
                                            )
                    
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                
                T2 = time.time()
                print('Running Time: %s s' % ((T2 - T1)))
                
                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    # img.save(os.path.join(sample_path, f"{base_count:05}_{prompts[0]}.png"))
                    return img

                    
def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'clicks': 0,
}

def get_select_coordinates(img, evt: gr.SelectData):
    sections = []
    # update new coordinates
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['y_temp'] = ROI_coordinates['y_new']
    ROI_coordinates['x_new'] = evt.index[0]
    ROI_coordinates['y_new'] = evt.index[1]
    # compare start end coordinates
    x_start = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_start = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    x_end = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_end = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    
    if ROI_coordinates['clicks'] % 2 == 0:
        # both start and end point get
        sections.append(((x_start, y_start, x_end, y_end), "Mask for composing object"))
        
        DICT_MASK['x_1'] = x_start
        DICT_MASK['y_1'] = y_start
        DICT_MASK['x_2'] = x_end
        DICT_MASK['y_2'] = y_end
        
        return (img, sections)
    else:
        # point_width = int(img.shape[0]*0.05)
        point_width = 20
        sections.append(((ROI_coordinates['x_new'], ROI_coordinates['y_new'], 
                          ROI_coordinates['x_new'] + point_width, ROI_coordinates['y_new'] + point_width),
                        "Click second point for mask"))
        return (img, sections)
    

css = '''
.container {max-width: 1150px;margin: auto;}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
'''
example={}
ref_dir='./gradio/foreground'
image_dir='./gradio/background'
seg_dir='./gradio/seg_foreground'
ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir)]
ref_list.sort()
image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir)]
image_list.sort()
seg_list=[os.path.join(seg_dir,file) for file in os.listdir(seg_dir)]
seg_list.sort()

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(read_content("./scripts/header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', elem_id="image_upload", type="pil", label="Background Image").style(height=400)                   
                    reference = gr.Image(source='upload', elem_id="image_upload", type="pil", label="Foreground Image").style(height=400)
                    
                    with gr.Row():
                    # guidance = gr.Slider(label="Guidance scale", value=5, maximum=15,interactive=True)
                        steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1,interactive=True)
                        seed = gr.Slider(0, 10000, label='Seed (0 = random)', value=3407, step=1)
                    
                    with gr.Row():
                        tau_a = gr.Slider(label="tau_a", value=0.4, minimum=0.0, maximum=1.0, step=0.1, interactive=True,
                                          info="Foreground Attention Injection")
                        tau_b = gr.Slider(label="tau_b", value=0.8, minimum=0.0, maximum=1.0, step=0.1, interactive=True,
                                          info="Background Preservation")
                        
                    with gr.Row():
                        scale = gr.Slider(label="CFG", value=2.5, minimum=0.0, maximum=15.0, step=0.5, interactive=True,
                                        info="CFG=2.5 for real domain CFG>=5.0 for cross domain")
                        dpm_order = gr.CheckboxGroup(["1", "2", "3"], value = '2', label="DPM Solver Order") 
                        
                    domain = gr.Radio(["Cross Domain", "Real Domain"], value='Real Domain', label="Domain",
                                      info="When background is real image, choose Real Domain; otherwise, choose Cross Domain")
                    prompt = gr.Textbox(label="Prompt", 
                                        info="an oil painting (or a pencil drawing) of a panda"
                                        ).style(height=400)
                    
                    btn = gr.Button("Run!").style(
                        margin=True,
                        rounded=(True, True, True, True),
                        full_width=False,
                    )
                    
                with gr.Column():    
                    mask = gr.AnnotatedImage(label="Composition Region", 
                                             info="Setting mask for composition region: first click for the top left corner, second click for the bottom right corner",
                                             color_map={"Region for Composing Object": "#9987FF",
                                                        "Click Second Point for Mask": "#f44336"}).style(height=400)

                    image.select(get_select_coordinates, image, mask) 
                    
                    seg = gr.Image(source='upload', elem_id="image_upload", type="pil", 
                                   label="Segmentation Mask for Foreground").style(height=400)
                                       
                    image_out = gr.Image(label="Output", elem_id="output-img").style(height=400)
                        
                    # with gr.Group(elem_id="share-btn-container"):
                    #     community_icon = gr.HTML(community_icon_html, visible=True)
                    #     loading_icon = gr.HTML(loading_icon_html, visible=True)
                    #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
            
            
            with gr.Row():
                with gr.Column():
                    gr.Examples(image_list, inputs=[image],label="Examples - Background Image", examples_per_page=12)
                with gr.Column():
                    gr.Examples(ref_list, inputs=[reference],label="Examples - Foreground Image", examples_per_page=12)
                with gr.Column():
                    gr.Examples(seg_list, inputs=[seg],label="Examples - Foreground Segmentation Map", examples_per_page=12)
            
            
            btn.click(fn=tficon, 
                      inputs=[image, reference, seg, prompt, dpm_order, steps, tau_a, tau_b, domain, seed, scale], 
                      outputs=[image_out])
            
            # share_button.click(None, [], [], _js=share_js)


        #     gr.HTML(
        #         """
        #             <div class="footer">
        #                 <p>Model by <a href="" style="text-decoration: underline;" target="_blank">Fantasy-Studio</a> - Gradio Demo by 🤗 Hugging Face
        #                 </p>
        #             </div>
        #             <div class="acknowledgments">
        #                 <p><h4>LICENSE</h4>
        # The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
        #         """
        #     )

if __name__ == '__main__':
    demo.launch(inbrowser=True)