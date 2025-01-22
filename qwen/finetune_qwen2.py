import os
import json
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig,PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, metadata_folder, buyer_images_folder, seller_images_folder, pixel_size=424):
        self.metadata_folder = metadata_folder
        self.buyer_images_folder = buyer_images_folder
        self.seller_images_folder = seller_images_folder
        self.data = self.load_metadata()
        self.width = pixel_size
        self.height = pixel_size
        self.image_transform = transforms.Compose([
            transforms.Resize((self.width, self.height)), 
        ])
    def load_metadata(self):
        data = []
        for file_name in os.listdir(self.metadata_folder):
            if not file_name.endswith('.json'):
                continue
            file_path = os.path.join(self.metadata_folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data.append(json.load(f))
            except Exception as e:
                print(f"Skipping file {file_name} due to error: {e}")
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images
        buyer_image_path = os.path.join(self.buyer_images_folder, os.path.basename(item.get("buyer_image", "")))
        try:
            buyer_image = Image.open(buyer_image_path).convert("RGB")
            buyer_image = self.image_transform(buyer_image)
        except Exception as e:
            print(f"buyer image exception for {idx} path: {buyer_image_path}, error: {e}")
            buyer_image = Image.new("RGB", (self.width, self.height))
        
        seller_image_path = os.path.join(self.seller_images_folder, os.path.basename(item.get("seller_image", "")))
        try:
            seller_image = Image.open(seller_image_path).convert("RGB")
            seller_image = self.image_transform(seller_image)
        except Exception as e:
            print(f"seller image exception for {idx} path: {seller_image_path}, error: {e}")
            seller_image = Image.new("RGB", (self.width, self.height))
        
        # get descriptions
        buyer_description = item.get("review", "") or ""
        seller_review = item.get("seller_description", "") or ""
        
        # get the label exactly as it is in the metadata
        label_text = item.get("label", "")
        if not isinstance(label_text, str):
            label_text = str(label_text)  # ensure label is a string
        expected_output = item.get("expected-output", "")
        # system message
        system_message = "You are an AI assistant helping with buyer and seller interactions."
        
        sample = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Buyer Description: {buyer_description}"},
                    {"type": "image", "image": buyer_image},
                    {"type": "text", "text": f"Seller Description: {seller_review}"},
                    {"type": "image", "image": seller_image},
                    {"type": "text", "text": "Given the seller imager, seller description, buyer image and buyer description, I want you to classify the new sample based on these classes:\n-1: Buyer's opinion\n0: Here seller is at fault\n1: Here buyer agreed with the seller\n"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Label:\n{label_text}\n\nReasoning:\n{expected_output}\n"}],
            },
        ]
        
        return sample

# load testing set
metadata_folder = "./datasplit_new/training/metadata/"
buyer_images_folder = "./datasplit_new/training/buyer_images/"
seller_images_folder = "./datasplit_new/training/seller_images/"
pixel_size = 424
dataset = CustomDataset(metadata_folder, buyer_images_folder, seller_images_folder, pixel_size)

print(dataset[7])
path = f"./models/reviewqwen_large_2"

train_size = int(0.85 * len(dataset))
val_size = int(len(dataset) - train_size)
#test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(train_dataset[0])

from transformers import BitsAndBytesConfig
model_id = "Qwen/Qwen2-VL-7B-Instruct"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# load model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
#    quantization_config=bnb_config
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
processor.image_processor.size = {'height': pixel_size, 'width': pixel_size}
processor.image_processor.do_resize = True

# define collate function
def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_tokens = [151652, 151653, 151655]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch

# PEFT configurations
peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.2,
    r=40, # 8 used lot of memory, try 4 next
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
#model = get_peft_model(model, peft_config)

logging_eval_save_steps = 50
save_steps = 100

# set up training args
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=8,
    per_device_train_batch_size=8, # icnresed from 1
    per_device_eval_batch_size=4,  # same
    weight_decay=0.05,
    gradient_accumulation_steps=2, # originally was 4
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    logging_steps=25, # increased
    eval_steps=25,    # both of these to reduce frequency & computational overhead
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=50,  # increased to reduce frequency
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    bf16=False,  # disable BF16
    fp16=True,   # enable FP16    
    #bf16=True,
    #tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=False,
    report_to=None,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
training_args.remove_unused_columns = False

# set up the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
    callbacks=[]
)

# start finetuning
trainer.train()
# save the finetuned model
trainer.save_model(path)
print(f"model saved to: {path}")
