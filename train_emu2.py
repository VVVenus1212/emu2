import cv2
import os
import numpy as np
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers import CLIPImageProcessor, PretrainedConfig
from model import EmuVisualGenerationModel, EMU_InputProcessor
from transformers import Trainer
from .dataloader import SupervisedDataset, DataCollator
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import json
from accelerate import Accelerator
from .emugenconfig import EMUGENConfig


training_args = TrainingArguments(output_dir="llm_appr_lr55_cake2048_autoregress_iani_ep50",per_device_train_batch_size=1,bf16=True,bf16_full_eval=True,num_train_epochs=50,save_strategy='epoch', save_total_limit=1,
                                  report_to='tensorboard',logging_strategy='steps',logging_first_step=True,logging_steps=1,deepspeed='deepspeed_zero3.json',learning_rate=5e-5,warmup_ratio=0.05)

with open('config.json','r') as f:
    config = json.load(f)
Aconfig = EMUGENConfig(**config)

Emumodel = EmuVisualGenerationModel(Aconfig)

input_processor = EMU_InputProcessor(Emumodel.tokenizer,Emumodel.transform)
train_dataset = SupervisedDataset(data_path='/mnt/neimeng/nlp/projects/pretrain/xingjin/MineDojo/2048_data_cupcakes/all_1-300_action+des.json',input_processor=input_processor, output_vis_processor=Emumodel.transform)
data_collator = DataCollator(Emumodel.tokenizer)


trainer = Trainer(
    model=Emumodel,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

