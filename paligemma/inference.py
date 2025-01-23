#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
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
# -- Replace with PaliGemma imports --
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)
def load_and_resize_image(image_path, pixel_size=424):
    try:
        if isinstance(image_path, Image.Image):
            image = image_path
            image = image.resize((pixel_size, pixel_size))
            return image
        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        
        image = image.resize((pixel_size, pixel_size))
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return Image.new("RGB", (pixel_size, pixel_size))

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image':
                    image_url = content.get('image', None)
                    if image_url:
                        image_inputs.append(image_url)
                elif content['type'] == 'video':
                    video_url = content.get('video', None)
                    if video_url:
                        video_inputs.append(video_url)
    return image_inputs, video_inputs

class CustomDataset(Dataset):
    def __init__(self, metadata_folder, buyer_images_folder, seller_images_folder, pixel_size=424):
        self.metadata_folder = metadata_folder
        self.buyer_images_folder = buyer_images_folder
        self.seller_images_folder = seller_images_folder
        self.data = self.load_metadata()
        self.pixel_size = pixel_size
        self.image_transform = transforms.Compose([
            transforms.Resize((self.pixel_size, self.pixel_size)),
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

        # Buyer / Seller image paths
        buyer_path = os.path.join(self.buyer_images_folder, os.path.basename(item.get("buyer_image", "")))
        seller_path = os.path.join(self.seller_images_folder, os.path.basename(item.get("seller_image", "")))

        # Buyer / Seller text
        buyer_text = item.get("review", "") or ""
        seller_text = item.get("seller_description", "") or ""
        label_text = item.get("label", "") or ""

        return {
            "system_text": "You are an AI assistant helping with buyer-seller interactions.",
            "buyer_text": buyer_text,
            "seller_text": seller_text,
            "buyer_image_path": buyer_path,
            "seller_image_path": seller_path,
            "label_text": label_text
        }
label_mapping = {
    "-1": "-1",
    "0": "0",
    "1": "1"
}

def parse_label(label_str):
    cleaned_label_str = " ".join(label_str.split())  

    # regex to match multiple formats for labels (-1:, 0:, 1:, or variations with Answer)
    label_regex = r"(?<!\d)(-1|0|1)|(?:\*Answer\*[:\s]*(-1|0|1))"


    # Search for a match
    match = re.search(label_regex, label_str)
    if match:
        # Extract the label from the appropriate group
        if match.group(1):  # Match for `-1:`, `0:`, `1:`
            extracted_label = match.group(1)
        elif match.group(2):  # Match for `*Answer*: -1`, `*Answer*: 0`, `*Answer*: 1`
            extracted_label = match.group(2)
        else:
            extracted_label = None

        #print("Extracted label:", extracted_label)
        return int(extracted_label) if extracted_label is not None else -37
    else:
        print("No valid label found.")
        return -37  # Default value for invalid labels

pixel_size = 424
model_id = "google/paligemma2-10b-pt-896"
model_name = model_id.split("/")[-1]
print(f"model name: {model_name}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #quantization_config=bnb_config,  # uncomment if needed
).eval()

processor = PaliGemmaProcessor.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_predictions(model, processor, batch, device):
    generated_texts = []

    for sample in batch:
        # load images
        buyer_img = load_and_resize_image(sample["buyer_image_path"], pixel_size=524)
        seller_img = load_and_resize_image(sample["seller_image_path"], pixel_size=524)
        images = [buyer_img, seller_img]

        prompt_text = (
            "<image><image> "  # exactly 2 <image> tokens
            + sample["system_text"] + "\n\n"
            + f"Buyer says: {sample['buyer_text']}\n"
            + f"Seller says: {sample['seller_text']}\n\n"
            "Classify this interaction:\n"
            "-1: Buyer's opinion\n"
            "0: Seller is at fault\n"
            "1: Buyer is satisfied\n"
        )
        inputs = processor(
            text=prompt_text,  # text prompt
            images=[images[0], images[1]],
            return_tensors="pt"
        )

        for key, val in inputs.items():
            if key == "pixel_values":
                inputs[key] = val.to(torch.bfloat16).to(device)
            else:
                inputs[key] = val.to(device)
        # generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True, 
                temperature=0.2,
                top_p=0.9
            )
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        # deccode the entire output to see if it’s coherent
        output_text = processor.decode(new_tokens, skip_special_tokens=True)
        print("Model Output:", output_text)

        generated_texts.append(output_text)

    return generated_texts

results = []
for batch in tqdm(dataloader, desc="Running Inference"):
    generated_texts = generate_predictions(model, processor, batch, device)
    for i, sample in enumerate(batch):
        #assistant = next((item for item in sample if item['role'] == 'assistant'), None)
        true_label_str = sample['label_text'].split()[0]#assistant['content'][0]['text'] if assistant else ""
        
        true_label = parse_label(true_label_str)
        generated_response = generated_texts[i]
        predicted_label = parse_label(generated_response)
        print(f"predicted label: {predicted_label}, true: {true_label}")
        results.append({
            "generated_response": generated_response,
            "predicted_label": predicted_label,
            "true_label": true_label
        })

with open(f"./testing/{model_name}_large_testing.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

true_labels = [item["true_label"] for item in results]
predictions = [item["predicted_label"] for item in results]
filtered_true, filtered_pred = [], []
for t, p in zip(true_labels, predictions):
    if t != -37 and p != -37:
        filtered_true.append(t)
        filtered_pred.append(p)

accuracy = accuracy_score(filtered_true, filtered_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

precision, recall, f1, _ = precision_recall_fscore_support(
    filtered_true, filtered_pred, average='weighted', zero_division=0
)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")

target_names = ["Negative", "Neutral/Fault", "Positive"]
print("\nClassification Report:")
print(classification_report(filtered_true, filtered_pred, target_names=target_names, zero_division=0))

