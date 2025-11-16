import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model, PeftModel

from .data import main as data_main
from .gpt_2_small import main as gpt_2_small_main
from .gpt_2_large import main as gpt_large_main
if __name__ == "__main__":
  # 1. Data
  print("-------------------------- Run data.py --------------------------\n")
  data_main()
  # 2. Gpt 2 small
  print("-------------------------- Run gpt_setup.py --------------------------\n")
  gpt_2_small_main()
  # 3. Gpt 2 large
  print("-------------------------- Run full_finetune.py --------------------------\n")
  gpt_2_large_main()
