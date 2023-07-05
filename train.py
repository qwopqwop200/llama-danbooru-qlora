#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install transformers')
#get_ipython().system('pip install bitsandbytes')
#get_ipython().system('pip install git+https://github.com/huggingface/peft')
#get_ipython().system('pip install datasets')
#get_ipython().system('pip install scipy')
#get_ipython().system('pip install sentencepiece')


# In[2]:


import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import os
import numpy as np


# In[3]:


np.random.seed(42)
model_id = "pinkmanlove/llama-7b-hf"
DEFAULT_PAD_TOKEN = "[PAD]"


# In[4]:
n_gpus = torch.cuda.device_count()
max_memory = f'{24000}MB'
max_memory = {i: max_memory for i in range(n_gpus)}
device_map = "auto"

# if we are in a distributed setting, we need to set the device map and max memory per device
if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    max_memory = {'': max_memory[local_rank]}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device_map, max_memory=max_memory)

setattr(model, 'model_parallel', True)
setattr(model, 'is_parallelizable', True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# In[5]:


tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

if tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model)

tokenizer.add_special_tokens({"eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                              "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                              "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id),})


# In[6]:


data = load_dataset('qwopqwop/danbooru2022_tags')['train']
data = data.map(lambda x: {'text': f"{', '.join(list(np.random.permutation(x['tags'].split(', '))))}{tokenizer.eos_token}" })
data = data.map(lambda samples: tokenizer(samples["text"],max_length=2048, truncation=True), batched=True)


# In[7]:


data[0]['tags']


# In[8]:


data[0]['text']


# In[9]:


data[0]['input_ids']


# In[10]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
config = LoraConfig(r=64,
                    lora_alpha=16,
                    target_modules=modules,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                   )
model.enable_input_require_grads()
model = get_peft_model(model, config)
print_trainable_parameters(model)


# In[11]:


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=10,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_32bit",
        lr_scheduler_type= "constant",
        ddp_find_unused_parameters=False
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.add_callback(SavePeftModelCallback)
trainer.train()

