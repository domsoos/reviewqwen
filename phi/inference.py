#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor
)


def load_and_resize_image(image_path, pixel_size=424):
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        image = image.resize((pixel_size, pixel_size))
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a blank image if failing
        return Image.new("RGB", (pixel_size, pixel_size))

def parse_true(label):
    label = label.split()[0]
    return label.split(":")[0]

def parse_label(label_str):
    import re
    
    cleaned_label = label_str.split()[0]# " ".join(label_str.split())  # remove extra whitespace
    match = re.search(r"(-1|0|1)\s*", cleaned_label)
    if match:
        return int(match.group(1))
    else:
        return -37

"""
def parse_true(true_label_str):
    #Extracts the ground-truth label (assumes format like '-1:some text').
    return true_label_str.split(':')[0]
"""

class CustomDataset(Dataset):
    def __init__(self, metadata_folder, buyer_images_folder, seller_images_folder, pixel_size=424):
        self.metadata_folder = metadata_folder
        self.buyer_images_folder = buyer_images_folder
        self.seller_images_folder = seller_images_folder
        self.data = self.load_metadata()
        self.pixel_size = pixel_size

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

        # Load buyer + seller images
        buyer_image_path = os.path.join(self.buyer_images_folder, os.path.basename(item.get("buyer_image", "")))
        seller_image_path = os.path.join(self.seller_images_folder, os.path.basename(item.get("seller_image", "")))
        buyer_image = load_and_resize_image(buyer_image_path, self.pixel_size)
        seller_image = load_and_resize_image(seller_image_path, self.pixel_size)

        # Buyer + Seller text
        buyer_description = item.get("review", "") or ""
        seller_description = item.get("seller_description", "") or ""

        # True label as string
        label_text = item.get("label", "")
        if not isinstance(label_text, str):
            label_text = str(label_text)

        # Return a dict with everything needed
        return {
            "buyer_image": buyer_image,
            "seller_image": seller_image,
            "buyer_text": buyer_description,
            "seller_text": seller_description,
            "label_text": label_text
        }


model_id = "microsoft/Phi-3.5-vision-instruct"
pixel_size = 524

print(f"Loading model: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",  # or "torch.float16"
    _attn_implementation='eager'
    #_attn_implementation='flash_attention_2'
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4
)

metadata_folder = "../datasplit_new/testing/metadata"
buyer_images_folder = "../datasplit_new/testing/buyer_images"
seller_images_folder = "../datasplit_new/testing/seller_images"

dataset = CustomDataset(
    metadata_folder=metadata_folder,
    buyer_images_folder=buyer_images_folder,
    seller_images_folder=seller_images_folder,
    pixel_size=pixel_size
)

def custom_collate_fn(batch):
    return batch

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)

def generate_predictions(model, processor, batch, device="cuda", max_new_tokens=128):
    """
    For each sample in `batch`, builds the prompt with 2 <|image|> placeholders,
    runs the model, then extracts the new tokens as output.
    """
    generated_texts = []

    for sample in batch:
        buyer_img = sample["buyer_image"]
        seller_img = sample["seller_image"]

        placeholders = "<|image_1|>\n<|image_2|>\n"

        # user prompt
        user_text = (
            f"{placeholders}"
            f"Buyer description: {sample['buyer_text']}\n"
            f"Seller description: {sample['seller_text']}\n"
            "Classify with only one of:\n"
            "-1: Buyer's opinion\n"
            " 0: Seller at fault\n"
            " 1: Buyer agreed\nYour classification and reasoning:\n"
        )

        messages = [
            {"role": "user", "content": user_text}
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            prompt,
            [buyer_img, seller_img],
            return_tensors="pt"
        ).to(device)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.8,
            "do_sample": False,
            "eos_token_id": processor.tokenizer.eos_token_id
        }

        # Generate
        with torch.no_grad():
            generate_ids = model.generate(**inputs, **generation_args)

        # Remove input tokens from generation
        input_len = inputs["input_ids"].shape[1]
        new_ids = generate_ids[:, input_len:]

        # Decode
        output_text = processor.batch_decode(
            #generate_ids,    
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(output_text)
        generated_texts.append(output_text)

    return generated_texts


results = []

from tqdm import tqdm

for batch in tqdm(dataloader, desc="Running Inference"):
    predictions = generate_predictions(model, processor, batch)
    for i, sample in enumerate(batch):
        # True label
        true_label_str = sample["label_text"]
        true_label = parse_true(true_label_str) 

        # Predicted label from model output
        predicted_label = parse_label(predictions[i])

        print(f"Predicted: {predicted_label}, True: {true_label}")
        results.append({
            "generated_response": predictions[i],
            "predicted_label": predicted_label,
            "true_label": true_label
        })

# Save results
with open("./testing/phi3.5_testing.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

filtered_results = []
for r in results:
    lbl = r["true_label"].strip()
    if lbl.isdigit() or (lbl.startswith('-') and lbl[1:].isdigit()):
        filtered_results.append(r)
    else:
        # skip or log a warning
        print(f"Skipping invalid label: '{lbl}'")

# Evaluate
true_labels = [int(r["true_label"]) for r in filtered_results]
pred_labels = [int(r["predicted_label"]) for r in filtered_results]

# Filter out invalid
filtered_true, filtered_pred = [], []
for t, p in zip(true_labels, pred_labels):
    if t != -37 and p != -37:
        filtered_true.append(t)
        filtered_pred.append(p)

acc = accuracy_score(filtered_true, filtered_pred)
print(f"Accuracy: {acc*100:.2f}%")

prec, rec, f1, _ = precision_recall_fscore_support(
    filtered_true, filtered_pred, average='weighted', zero_division=0
)
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(filtered_true, filtered_pred, zero_division=0))

