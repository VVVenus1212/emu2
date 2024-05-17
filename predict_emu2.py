import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,AutoModel
import torch
from transformers import CLIPImageProcessor, PretrainedConfig
from model import EmuVisualGenerationModel, EMU_InputProcessor
from transformers import Trainer
from .dataloader import SupervisedDataset, DataCollator
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import json
from accelerate import Accelerator
import torch
import random


path = "/mnt/neimeng/nlp/projects/pretrain/xingjin/Emu/Emu2/checkpoint"
path_save = "/mnt/neimeng/nlp/projects/pretrain/xingjin/Emu/Emu2/test_res3/"
# path_pic = "/mnt/neimeng/nlp/projects/pretrain/xingjin/Emu/Emu2/examples/"
path_pic = "/mnt/neimeng/nlp/projects/pretrain/xingjin/MineDojo/2048_data1/data_0/"



with open('/mnt/neimeng/nlp/projects/pretrain/xingjin/MineDojo/2048_data_cupcakes/all_1-300_action+des.json','r') as f:
    new_data = json.load(f)

random_selection = new_data[0:100]

Emumodel = EmuVisualGenerationModel.from_pretrained("/mnt/neimeng/nlp/projects/pretrain/xingjin/Emu/llm_appr_lr15_cake2048_autoregress_iani_ep50/checkpoint-1218/",device_map='cuda:0')
# Emumodel = EmuVisualGenerationModel.from_pretrained("/mnt/neimeng/nlp/projects/pretrain/xingjin/Emu/llm_appr_lr15_cake2048_autoregress_pap_ep50/checkpoint-512/",device_map='cuda:0')

for i in range(50):
    str_now = random_selection[i][1]["Action"]
    # str_now ='1-1:yellow cake 1-2:dark cake 1-3:blank 1-4:white cake 2-1:brown cake 2-2:green cake 2-3:yellow cake 2-4:blank 3-1:pink cake 3-2:dark cake 3-3:brown cake 3-4:white cake 4-1:pink cake 4-2:yellow cake 4-3:white cake 4-4:blank '
    # str_now = ''
    image_now = Image.open(random_selection[i][0]["pic"]).convert('RGB')
    # image_now1 = Image.open(random_selection[i+1][0]["pic"]).convert('RGB')
    prompt = [image_now, str_now]
    # prompt = [str_now]
    ret = Emumodel.generate_images(prompt)
    ret.image.save(path_save + f"{i}.png")

    image = Image.open(path_save + f"{i}.png")

    # 创建一个可编辑的图片副本
    draw = ImageDraw.Draw(image)

    # 定义要添加的文本和文本颜色
    text = str_now
    # text_color = (255, 255, 255)  # 白色

    # 选择字体和字体大小
    # font = ImageFont.truetype("arial.ttf", 36)

    # 确定文本位置
    text_position = (10, 10)

    # 在图片上添加文本
    draw.text(text_position, text)

    # 保存修改后的图片
    image.save(path_save + f"{i}.png")


print('a')



