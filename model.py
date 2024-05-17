# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-19 10:45
#    Last Modified : 2023-12-25 07:59
#    File Name     : pipeline_emu2_gen.py
#    Description   :
#
# ===========================================================================================

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as TF
from tqdm import tqdm
from typing import Any, Optional, Dict, List
import random
import torch.nn.functional as F

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.models.vae import DiagonalGaussianDistribution

from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from Emu2.checkpoint.multimodal_encoder.modeling_emu import EmuForCausalLM,USE_LORA
from Emu2.checkpoint.multimodal_encoder.constants import *
from emugenconfig import EMUGENConfig

from peft import (
    LoraConfig,
    PeftType,
    get_peft_model,
    PrefixTuningConfig,
    PromptLearningConfig,
    PrefixEncoder,
    TaskType
)

EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
USE_CFG = False
USE_SNR = False


def trainable_para(mymodel):
    total_params = 0
    for name, param in mymodel.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"Parameter name: {name}, Size: {param.numel()}")
    print(f"Total size: {total_params}")


class EMU_InputProcessor(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, image_processor: Any):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __call__(self, text = None, images = None, **kwargs) -> Any:
        output_dict = {}
        if text is not None:
            text_output = self.tokenizer(text,padding="longest", return_tensors="pt",add_special_tokens=False)
            output_dict.update(text_output)

        if images is not None:
            all_images = []
            if isinstance(images, list):
                for img in images:
                    image_output = self.image_processor(img)
                    all_images.append(image_output)
                input_images = torch.stack(all_images, dim=0)
            else:
                input_images = self.image_processor(images)
            output_dict['input_images'] = input_images
        return output_dict
    

@dataclass
class EmuVisualGenerationPipelineOutput(BaseOutput):
    image: Image.Image
    nsfw_content_detected: Optional[bool]


class EmuVisualGenerationModel(PreTrainedModel):
    config_class = EMUGENConfig

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer_path = config.tokenizer
        self.multimodal_encoder_path = config.multimodal_encoder
        self.scheduler_path =config.scheduler
        self.unet_path =config.unet
        self.vae_path =config.vae
        self.feature_extractor_path =config.feature_extractor
        self.safety_checker_path =config.safety_checker

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.multimodal_encoder = EmuForCausalLM.from_pretrained(
            self.multimodal_encoder_path,
            # config = multimodal_config
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            # device_map="auto"
           )
        
        # print('load multimodal_encoder')
        
        if USE_LORA:
            print("Using LoRA")
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # modules_to_save=['lm_head','embed_tokens']
            )
            self.multimodal_encoder.model.decoder.lm = get_peft_model(self.multimodal_encoder.model.decoder.lm, self.lora_config)
            self.multimodal_encoder.model.decoder.lm.print_trainable_parameters()

        # self.multimodal_encoder.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(self.unet_path,torch_dtype=torch.bfloat16,use_safetensors=True,variant="bf16",
                                                        #  device_map="auto"
                                                         )
        # print('load unet')
        self.vae = AutoencoderKL.from_pretrained(self.vae_path,torch_dtype=torch.bfloat16,use_safetensors=True,variant="bf16", 
                                                #  device_map="auto"
                                                 )
        # print('load vae')
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.scheduler_path,torch_dtype=torch.bfloat16,use_safetensors=True,variant="bf16")
        # print('load scheduler')
        self.feature_extractor = CLIPImageProcessor.from_pretrained(self.feature_extractor_path,torch_dtype=torch.bfloat16,use_safetensors=True,variant="bf16")
        # print('load feature_extractor')
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(self.safety_checker_path,torch_dtype=torch.bfloat16,use_safetensors=True,variant="bf16")
        # print('load safety_checker')

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.scheduler.requires_grad_(False)

        # trainable_para(self.multimodal_encoder)
        

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.transform = TF.Compose([
            TF.Resize((EVA_IMAGE_SIZE, EVA_IMAGE_SIZE), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])

        self.negative_prompt = {}

        self.output_img_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMG_TOKEN)
        self.img_token_num = 64
        hidden_size = self.multimodal_encoder.model.decoder.lm.model.config.hidden_size
        self.zero_img_feature = torch.zeros((1, self.img_token_num, hidden_size)).to(torch.bfloat16)

    def device(self, module):
        return next(module.parameters()).device

    def dtype(self, module):
        return next(module.parameters()).dtype
    
    def input_warp(self, input_ids, attention_mask, labels=None, input_image=None, input_image_feature=None, output_image=None):
        assert input_ids.shape[0] == 1, "warping each sample individually"

        IMAGE, BOI = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        
        if input_image is not None:
            device = self.multimodal_encoder.model.visual.patch_embed.proj.weight.device
            input_image = input_image.to(device)
            prompt_image_embeds = self.multimodal_encoder.model.encode_image(input_image)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            device = self.multimodal_encoder.project_up.weight.device
            prompt_image_embeds = prompt_image_embeds.to(device)
            prompt_image_embeds = self.multimodal_encoder.project_up(prompt_image_embeds)
        
        if output_image is not None:
            device = self.multimodal_encoder.model.visual.patch_embed.proj.weight.device
            output_image = output_image.to(device)
            out_prompt_image_embeds = self.multimodal_encoder.model.encode_image(output_image)
            _, _, c = out_prompt_image_embeds.shape
            out_prompt_image_embeds = out_prompt_image_embeds.view(-1, c)
            device = self.multimodal_encoder.project_up.weight.device
            out_prompt_image_embeds = out_prompt_image_embeds.to(device)
            out_prompt_image_embeds = self.multimodal_encoder.project_up(out_prompt_image_embeds)
        
        device = self.multimodal_encoder.model.decoder.lm.model.device
        attention_mask = attention_mask.to(device)
        input_ids = input_ids.to(device) # B x N

        if USE_LORA:
            text_embeds = self.multimodal_encoder.model.decoder.lm.base_model.model.model.embed_tokens(input_ids)
        else:
            text_embeds = self.multimodal_encoder.model.decoder.lm.model.embed_tokens(input_ids)
        image_idx = (input_ids == IMAGE)
        text_embeds[image_idx][:-64,:] = prompt_image_embeds.to(text_embeds.device)
        text_embeds[image_idx][-64:,:] = out_prompt_image_embeds.to(text_embeds.device)

        return text_embeds, attention_mask, labels

    def forward(
        self, input_ids, attention_mask, input_images, output_image, labels, captions=None, input_images_feature=None, output_image_feature=None, input_images_num = None
    ):
        batch_size = input_ids.shape[0]
        all_input_embeds, all_attention, all_labels,all_output_images = [], [], [], []
        for b in range(batch_size):
            if input_images_feature is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_image_feature=input_images_feature[b],output_image=output_image[b].unsqueeze(0))
            elif input_images is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_images[b][:input_images_num[b]],output_image=output_image[b].unsqueeze(0))
            wrapped_out_img_embeds = self.multimodal_encoder.model.encode_image(output_image[b].unsqueeze(0))
            # wrapped_inp_img_embeds = self.multimodal_encoder.model.encode_image(input_images[b][:input_images_num[b]])

            # loss_input_output = F.mse_loss(wrapped_out_img_embeds, wrapped_inp_img_embeds)
            # print(f'input_output_loss{loss_input_output}')


            all_input_embeds.append(wrapped_img_embeds)
            all_attention.append(wrapped_atts_img)
            all_labels.append(wrapped_labels)
            all_output_images.append(wrapped_out_img_embeds)

        #add padding features for batch
        # max_len = max([x.shape[1] for x in all_input_embeds])
        # for i in range(len(all_input_embeds)):
        #     if all_input_embeds[i].shape[1] < max_len:
        #         pad_len = max_len - all_input_embeds[i].shape[1]
        #         pad_embeds = torch.zeros([all_input_embeds[i].shape[0], pad_len, all_input_embeds[i].shape[2]]).to(all_input_embeds[i].device)
        #         pad_atts = torch.zeros([all_attention[i].shape[0], pad_len]).to(all_attention[i].device)
        #         pad_labels = torch.ones([all_labels[i].shape[0], pad_len], dtype=torch.long).to(all_labels[i].device) * -100
        #         all_input_embeds[i] = torch.cat([all_input_embeds[i], pad_embeds], dim=1)
        #         all_attention[i] = torch.cat([all_attention[i], pad_atts], dim=1)
        #         all_labels[i] = torch.cat([all_labels[i], pad_labels], dim=1)
        device = self.multimodal_encoder.model.decoder.lm.device 
        all_input_embeds = torch.cat(all_input_embeds, dim=0).to(device)
        all_attention = torch.cat(all_attention, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)
        all_output_images = torch.cat(all_output_images,dim=0).to(device)

        # past_key_values = None
        # if self.using_prefix_tuning:
        #     device = all_input_embeds.device
        #     past_key_values = self.get_prompt(batch_size=batch_size, device=device)
        #     prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(device)
        #     all_attention = torch.cat([prefix_attention_mask, all_attention], dim=1)
        #     # prefix_labels = torch.ones(batch_size, self.peft_config.num_virtual_tokens, dtype=wrapped_labels.dtype).to(device) * -100
        #     # wrapped_labels = torch.cat([prefix_labels, wrapped_labels], dim=1)

        outputs = self.multimodal_encoder.model.decoder.lm(
                inputs_embeds=all_input_embeds,
                attention_mask=all_attention,
                return_dict=True,
                labels=all_labels,
                output_hidden_states=True,
            )
        output_token_index = (all_labels == self.output_img_id).nonzero()

        if len(output_token_index):
            addon_index = torch.ones_like(output_token_index)*(-1)
            addon_index[:, 0] = 0
            output_token_index += addon_index

        text_loss = outputs['loss']
        last_hidden_state = outputs['hidden_states'][-1]
        t2i_input_embedding = []
        caption_feature = []
        calculate_caption_loss = None
        for i in range(len(output_token_index)):
            bs_id, seq_id = output_token_index[i]
            # random set 10% data with empty text feature
            if USE_CFG and random.random() < 0.1:
                t2i_input_embedding.append(self.zero_img_feature.to(last_hidden_state.device))
            else:
                t2i_input_embedding.append(last_hidden_state[bs_id:bs_id+1, seq_id:seq_id+self.img_token_num, :])

        if len(t2i_input_embedding) == 0:
            loss = 0.2 * text_loss
        
        else:
            device = self.multimodal_encoder.project_down.weight.device
            t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0).to(device)
            mapping_feature = self.multimodal_encoder.project_down(t2i_input_embedding)
            

            # if output_image_feature is None:
            #     image_loss = self.compute_image_loss(mapping_feature, output_image[output_token_index[:, 0]])
            # else:
            #     image_loss = self.compute_image_loss(mapping_feature, None, output_image_feature=output_image_feature[output_token_index[:, 0]])
            # print(f'mapping_feature_shape:{mapping_feature.shape}  output_image_embeds_shape:{all_output_images.shape}')
            image_loss = F.mse_loss(mapping_feature, all_output_images)
        
            if calculate_caption_loss:
                caption_feature = torch.cat(caption_feature, dim=0)
                caption_loss = F.mse_loss(mapping_feature, caption_feature)

                loss = 0.2 * text_loss + image_loss + 0.1 * caption_loss
            else:
                # loss = text_loss + image_loss
                # print(f'text_loss:{text_loss}  image_loss:{image_loss}')
                loss = image_loss
                # print('Only image loss')
        
        return ([loss])

    def compute_image_loss(self, mapping_feature, output_image, output_image_feature=None):
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor

        if output_image_feature is not None:
            latents = DiagonalGaussianDistribution(output_image_feature).sample()
        else:
            if len(output_image.shape) == 3:
                output_image = output_image.unsqueeze(0)
            
            device = self.vae.device
            output_image = output_image.to(device)
            latents = self.vae.encode(output_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        target = noise

        device = self.unet.device
        unet_added_conditions = {}
        time_ids = torch.LongTensor([1024,1024] + [0,0] + [height, width]).to(device).repeat(bsz)
        unet_added_conditions["time_ids"] = time_ids
        # if USE_CFG:
        #     unet_added_conditions["time_ids"] = torch.cat([time_ids, time_ids], dim=0)
        # else:
        #     unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(mapping_feature, dim=1).to(device)

        
        noisy_latents = noisy_latents.to(device)
        mapping_feature = mapping_feature.to(device)
        timesteps = timesteps.to(device)

        model_pred = self.unet(noisy_latents, timesteps, mapping_feature, added_cond_kwargs=unet_added_conditions).sample

        if USE_SNR:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def compute_snr(self,timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr



    @torch.no_grad()
    def generate_images(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.device(self.unet)
        dtype = self.dtype(self.unet)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
        ).to(dtype).to(device)
        batch_size = prompt_embeds.shape[0] // 2 if do_classifier_free_guidance else prompt_embeds.shape[0]

        unet_added_conditions = {}
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(device)
        if do_classifier_free_guidance:
            unet_added_conditions["time_ids"] = torch.cat([time_ids, time_ids], dim=0)
        else:
            unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(prompt_embeds, dim=1)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Post-processing
        images = self.decode_latents(latents)

        # 6. Run safety checker
        images, has_nsfw_concept = self.run_safety_checker(images)

        # 7. Convert to PIL
        images = self.numpy_to_pil(images)
        return EmuVisualGenerationPipelineOutput(
            image=images[0],
            nsfw_content_detected=None if has_nsfw_concept is None else has_nsfw_concept[0],
        )

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        device = self.device(self.multimodal_encoder.model.visual)
        dtype = self.dtype(self.multimodal_encoder.model.visual)

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.multimodal_encoder.model.encode_image(image=image_prompt)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in self.negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    self.negative_prompt[key] = self.multimodal_encoder.model.encode_image(image=negative_image)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)
        else:
            prompt = self.multimodal_encoder.generate_image(text=[text_prompt], image=image_prompt, tokenizer=self.tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in self.negative_prompt:
                    self.negative_prompt[key] = self.multimodal_encoder.generate_image(text=[""], tokenizer=self.tokenizer)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)

        return prompt

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(self, images: np.ndarray):
        if self.safety_checker is not None:
            device = self.device(self.safety_checker)
            dtype = self.dtype(self.safety_checker)
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(images), return_tensors="pt").to(device)
            images, has_nsfw_concept = self.safety_checker(
                images=images, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return images, has_nsfw_concept
