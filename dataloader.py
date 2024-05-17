from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
import transformers
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import re
import os
import numpy as np
from PIL import Image
import random
from pathlib import Path
import json
import copy
import torch
import os
import json
from Emu2.checkpoint.multimodal_encoder.constants import *
from transformers import DataCollatorWithPadding
PRECISION = torch.bfloat16
IMG_TOKEN_NUM = 64
ALL_IMG_TOKENS_STR = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * IMG_TOKEN_NUM + DEFAULT_IMG_END_TOKEN


class CC3MDataset(Dataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False):
        self.test = test

    def recover_data(self, saved_file):
        all_data = torch.load(saved_file)
        self.sources = all_data['sources']
        self.targets = all_data['targets']
        self.input_image_path = all_data['input_image_path']
        self.output_image_path = all_data['output_image_path']
        self.caption = all_data['caption']
        self.task_names = all_data['task_names']
        del all_data
        if self.test:
            self.valid_idx = []
            for i in range(len(self.targets)):
                if self.output_image_path[i] is not None:
                    self.valid_idx.append(i)
            

    def save_process_data(self, saved_file):
        all_data = {'sources': self.sources,
                    'targets': self.targets,
                    'input_image_path': self.input_image_path,
                    'output_image_path': self.output_image_path,
                    'caption': self.caption,
                    'task_names': self.task_names,
                    }
        torch.save(all_data, saved_file)
    
    def __len__(self):
        if self.test:
            return len(self.valid_idx)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.test:
            i = self.valid_idx[i]
        input_image_path = self.input_image_path[i]
        output_image_path = self.output_image_path[i]
        input_text = self.sources[i]
        input_text = input_text.replace(DEFAULT_IMG_PLACEHOLDER,ALL_IMG_TOKENS_STR)
        output_text = self.targets[i]
        input_images = []
        for in_img_path in input_image_path:
            if in_img_path is not None:
                input_image = Image.open(in_img_path).convert("RGB")
            else:
                input_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            input_images.append(input_image)
        input_dict = self.input_processor(text = input_text, images = input_images, add_special_tokens=False)
        pad_image_num = 15-self.inimages_num[i]
        pad_image = torch.zeros_like(input_dict['input_images'][0]).unsqueeze(0).repeat(pad_image_num,1,1,1)
        input_dict['input_images'] = torch.cat([input_dict['input_images'],pad_image],dim=0)
        input_dict['input_images_num'] = torch.tensor(self.inimages_num[i]).view(1,-1)

        input_dict['original_images'] = input_images

        if output_image_path is not None:
            output_image = Image.open(output_image_path).convert("RGB")
            # output_image = self.expand2square(output_image, (255, 255, 255))
            output_image = self.output_vis_processor(output_image)
            output_image = output_image.unsqueeze(0)
        else:
            output_image_path = 'none'
            output_image = torch.zeros((1, 3, 512, 512))
        input_dict["output_image"] = output_image
                
        input_dict["caption"] = self.caption[i]
        # input_dict["task_name"] = self.task_names[i]
        target_ids = self.input_processor(text = output_text, add_special_tokens=False)['input_ids']
        # mask = torch.ge(target_ids, 32003).float()
        # target_ids = torch.where(mask == 1, -100, target_ids)
        label = torch.ones_like(input_dict["input_ids"])*-100
        label = torch.cat((label, target_ids), dim=1)
        index = torch.nonzero(label == self.BOI)
        if len(index):
            index = index[0,1]
            label[:, index+1:] = -100
        input_dict["labels"] = label
        input_dict["input_ids"] = torch.cat((input_dict["input_ids"], target_ids), dim=1)
        input_dict["attention_mask"] = torch.cat((input_dict["attention_mask"], torch.ones_like(target_ids)), dim=1)
        input_dict["attention_ids"] = input_text
        input_dict["target_ids"] = output_text

        return input_dict

    def pre_caption(self, caption):
        
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        # max_words = 100
        # caption_words = caption.split(" ")
        # if len(caption_words) > max_words:
        #     caption = " ".join(caption_words[: max_words])

        return caption
    
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class MineDataset(CC3MDataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False):
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.IMAGE, self.BOI = input_processor.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        self.sources, self.targets, self.input_image_path, self.output_image_path, self.inimages_num = [], [], [], [], []
        self.caption, self.task_names = [], []
        all_tasks = json.load(open(data_path, 'r'))
    
        # pic_path = '/mnt/neimeng/nlp/projects/pretrain/xingjin/MineDojo/'
        for task in tqdm(all_tasks):
            task_input_image_path = []

            tmp_prompt = DEFAULT_IMG_PLACEHOLDER
            # tmp_prompt = ''
            for step in task:
                for key,value in step.items():
                    if key == 'pic':
                        # value = pic_path + value
                        if len(task_input_image_path) == 0:
                            task_input_image_path.append(value)
                        else:
                            
                            self.sources.append(tmp_prompt)
                            self.targets.append(tmp_target)
                            self.caption.append(None)
                            # self.task_names.append(step_name+"-multimodal")
                            step_input_image = copy.deepcopy(task_input_image_path)
                            self.input_image_path.append(step_input_image)
                            self.inimages_num.append(len(step_input_image))
                            self.output_image_path.append(value)

                            task_input_image_path.append(value)
                            tmp_prompt += last_text
                            tmp_prompt += DEFAULT_IMG_PLACEHOLDER

                    elif key == 'Action':
                        tmp_prompt += value 
                    
                    elif key == 'Reason':
                        # tmp_target = f"{value} {ALL_IMG_TOKENS_STR}"
                        tmp_target = ALL_IMG_TOKENS_STR
                        last_text = value
                        # tmp_prompt += value
                        

        self.valid_idx = list(range(len(self.sources)))
        print('Load data done!')

    
@dataclass
class DataCollator(DataCollatorWithPadding):

    def __call__(self, instances):
        key_list = instances[0].keys()
        output_dict = {}
        for key in key_list:
            # Need to remove the batch dimension
            if key in ['input_ids', 'attention_mask', 'labels']:
                output_value = [instance[key][0] for instance in instances]
            elif key == 'input_images':
                output_value = [instance[key].unsqueeze(0) for instance in instances]
            else:
                output_value = [instance[key] for instance in instances]

            if key == "input_ids":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif key == "labels":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=-100)
            elif key == "attention_mask":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=0)
            elif key == 'input_images':
                output_value = [v.to(PRECISION) for v in output_value]
                output_value = torch.concat(output_value).to(PRECISION)
            elif key == 'output_image':
                output_value = torch.concat(output_value).to(PRECISION)
            elif key == 'output_image_feature':
                output_value = torch.concat(output_value)
            elif key == 'input_images_num':
                output_value = torch.concat(output_value)
            output_dict[key] = output_value
        return output_dict

SupervisedDataset = MineDataset
