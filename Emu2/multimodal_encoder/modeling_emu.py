from functools import partial
from typing import List, Optional
from argparse import Namespace
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .configuration_emu import EmuConfig
from .constants import *
from .modeling_llama import LlamaForCausalLM
from .visual import EVAVisionTransformer

from transformers.generation import StoppingCriteriaList,StoppingCriteria


USE_LORA = False
DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"

class EmuPreTrainedModel(PreTrainedModel):
    config_class = EmuConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer", "Block"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class EmuForClsAndRegression(EmuPreTrainedModel):

    def __init__(self, config):
        super(EmuForClsAndRegression, self).__init__(config)

        self.lm = LlamaForCausalLM(config=config)

        self.lm.model.embed_tokens.padding_idx = config.pad_token_id

    def get_num_layers(self):
        return len(self.lm.model.layers)

class EmuModel(EmuPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        vision_config = Namespace(**config.vision_config)

        self.visual = EVAVisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.width // vision_config.head_width,
            mlp_ratio=vision_config.mlp_ratio,
            qkv_bias=vision_config.qkv_bias,
            drop_path_rate=vision_config.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=vision_config.layer_norm_eps),
            xattn=vision_config.xattn,
            postnorm=vision_config.postnorm,
        )

        self.visual.requires_grad_(False)

        self.decoder = EmuForClsAndRegression(config)

        # self.decoder.requires_grad_(False)

        self.gradient_checkpointing = False
        
        self.n_query = vision_config.n_query
        self.v_query = vision_config.v_query

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, *, n_query=None):
        n_query = n_query if n_query is not None else self.n_query

        image_embeds = self.visual(image)
        image_embeds = image_embeds[:, 1:, :]
        b, n, c = image_embeds.shape
        sqrt_n = int(n**0.5)
        image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)

        stride = int(sqrt_n // (n_query ** 0.5))
        image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
        image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
        return image_embeds


class EmuForCausalLM(EmuPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model = EmuModel(config)
        # print('load encoder model')
        # LM to EVA
        self.project_down = nn.Linear(config.hidden_size, config.d_model, bias=False)
        # self.project_down.requires_grad_(False)
        # EVA to LMs
        self.project_up = nn.Linear(config.d_model, config.hidden_size, bias=False)
        # self.project_up.requires_grad_(False)
        

        self.n_query = self.model.n_query
        self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * self.n_query + DEFAULT_IMG_END_TOKEN

    def device(self, module=None):
        if module is None:
            return next(self.parameters()).device
        return next(module.parameters()).device

    def dtype(self, module):
        if module is None:
            return next(self.parameters()).dtype
        return next(module.parameters()).dtype

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: PreTrainedTokenizer,
        image: Optional[torch.Tensor] = None,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        IMAGE, BOI = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        out_str = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * 64
        if image is not None:
            prompt_image_embeds = self.model.encode_image(image)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.project_up(prompt_image_embeds)

            text = [t.replace(placeholder, self.image_placeholder) for t in text]
            # inputs = tokenizer(text, padding="longest", return_tensors="pt")
            # if USE_LORA:
            #     device = self.device(self.model.decoder.lm.base_model.model.model.embed_tokens)
            # else:
            #     device = self.device(self.model.decoder.lm.model.embed_tokens)
            # attention_mask = inputs.attention_mask.to(device)
            # input_ids = inputs.input_ids.to(device) # B x N
            # if USE_LORA:
            #     text_embeds = self.model.decoder.lm.base_model.model.model.embed_tokens(input_ids)
            # else:
            #     text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)
            
            # image_idx = (input_ids == IMAGE)
            # text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)

            # outputs = self.model.decoder.lm.model.generate(
            #         inputs_embeds=text_embeds,
            #         temperature = 0.8,
            #         # do_sample= True,
            #         # early_stopping=True,
            #         # stopping_criteria= DEFAULT_IMAGE_TOKEN,
            #         max_new_tokens = 100,
            #         # output_hidden_states = True,
            #     )
            
            # text_out = tokenizer.decode(outputs[0])
            # text[0] = text[0] + text_out.split(DEFAULT_IMG_TOKEN)[0]

        target_image_embeds = None
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                text = [f"{t}{DEFAULT_IMG_TOKEN}" for t in text]
            else:
                text = [f"{t}{DEFAULT_IMAGE_TOKEN}" for t in text]

            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            if USE_LORA:
                device = self.device(self.model.decoder.lm.base_model.model.model.embed_tokens)
            else:
                device = self.device(self.model.decoder.lm.model.embed_tokens)
            # device = self.device(self.model.decoder.lm.model.embed_tokens)
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device) # B x N

            if USE_LORA:
                text_embeds = self.model.decoder.lm.base_model.model.model.embed_tokens(input_ids)
            else:
                text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)


            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds.to(text_embeds.device)

            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                text_embeds[target_idx] = self.project_up(target_image_embeds).to(text_embeds.device)
                # text_embeds[target_idx] = target_image_embeds.to(text_embeds.device)

            outputs = self.model.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token+1))

            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[target_idx.to(hidden_states.device)]
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = self.project_down(target_image_embeds)

        # target_image_embeds = self.project_down(target_image_embeds)
        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        # return target_image_embeds
    
        # IMAGE, BOI = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        # out_str = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * 64
        # if image is not None:
        #     prompt_image_embeds = self.model.encode_image(image)
        #     _, _, c = prompt_image_embeds.shape
        #     prompt_image_embeds = prompt_image_embeds.view(-1, c)
        #     prompt_image_embeds = self.project_up(prompt_image_embeds)

        #     text = [t.replace(placeholder, self.image_placeholder)+out_str for t in text]
        #     inputs = tokenizer(text, padding="longest", return_tensors="pt")
        #     if USE_LORA:
        #         device = self.device(self.model.decoder.lm.base_model.model.model.embed_tokens)
        #     else:
        #         device = self.device(self.model.decoder.lm.model.embed_tokens)
        #     attention_mask = inputs.attention_mask.to(device)
        #     input_ids = inputs.input_ids.to(device) # B x N
        #     if USE_LORA:
        #         text_embeds = self.model.decoder.lm.base_model.model.model.embed_tokens(input_ids)
        #     else:
        #         text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)
            
        #     image_idx = (input_ids == IMAGE)
        #     image_idx[:,-64:] = False
        #     text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)
        #     # image_idx[:,:-64] = False
        #     # image_idx[:,-64:] = True
        #     # text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)

        #     outputs = self.model.decoder.lm.model(
        #         inputs_embeds=text_embeds,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )

        #     hidden_states = outputs.hidden_states[-1]
        #     target_image_embeds = hidden_states[:,-64:,:]
        #     target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
        #     target_image_embeds = self.project_down(target_image_embeds)

        #     _, C = target_image_embeds.shape
        #     B = hidden_states.shape[0]
        #     target_image_embeds = target_image_embeds.view(B, -1, C)
        
        
        # else:
        #     prompt_image_embeds = self.model.encode_image(image)
        #     _, _, c = prompt_image_embeds.shape
        #     prompt_image_embeds = prompt_image_embeds.view(-1, c)
        #     prompt_image_embeds = self.project_up(prompt_image_embeds)

        #     outputs = self.model.decoder.lm.model(
        #         inputs_embeds=prompt_image_embeds,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )

        #     hidden_states = outputs.hidden_states[-1]
        #     target_image_embeds = hidden_states[:,-64:,:]
        #     target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
        #     target_image_embeds = self.project_down(target_image_embeds)

        #     _, C = target_image_embeds.shape
        #     B = hidden_states.shape[0]
        #     target_image_embeds = target_image_embeds.view(B, -1, C)
        
        return target_image_embeds

