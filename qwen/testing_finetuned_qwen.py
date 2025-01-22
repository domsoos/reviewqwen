import os
import json
from torch.utils.data import Dataset, DataLoader
#from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re
import requests
from io import BytesIO
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
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
        return Image.new("RGB", (pixel_size, pixel_size))

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []

    for message in messages:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image':
                    image_url = content['image']
                    image_inputs.append(image_url)  # could be URL or local path
                elif content['type'] == 'video':
                    video_url = content['video']
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
            transforms.Resize((self.pixel_size, self.pixel_size)),  # resize images to specified pixel size
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
        buyer_image = load_and_resize_image(buyer_image_path, self.pixel_size)

        seller_image_path = os.path.join(self.seller_images_folder, os.path.basename(item.get("seller_image", "")))
        seller_image = load_and_resize_image(seller_image_path, self.pixel_size)

        # descriptions
        buyer_description = item.get("review", "") or ""
        seller_review = item.get("seller_description", "") or ""

        # label
        label_text = item.get("label", "")
        if not isinstance(label_text, str):
            label_text = str(label_text)  # ensure label is a string
        
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
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label_text}],
            },
        ]

        return sample

label_mapping = {
    "-1": "-1",  # Negative (Buyer's opinion)
    "0": "0", # Neutral/Fault (Seller at fault)
    "1": "1"    # Positive (Buyer agreed)
}

def parse_label(label_str):
    cleaned_label_str = " ".join(label_str.split())

    # simple regex: look for -1, 0, or 1 anywhere in the text
    label_regex = r"(-1|0|1)\s*"

    match = re.search(label_regex, cleaned_label_str)
    if match:
        extracted_label = match.group(1)
        #print("Extracted label:", extracted_label)  # Debugging
        return int(extracted_label)
    else:
        print("No valid label found.")
        return -37


# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

import sys

if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the model.")

path = sys.argv[1]
print(f"\npath 2 file: {path}")

pixel_size = 424#int(path.split("_")[2])
print(f"pixelsize: {pixel_size}")
model_id = "Qwen/Qwen2-VL-7B-Instruct"
model_name = path.split("/")[1]

print(f"model name: {model_name}")

finetuned_model_path = path
# uncomment the following line to test the base Qwen model
#finetuned_model_path = model_id

# Load the base model with quantization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    finetuned_model_path,
    #model_id,
    torch_dtype=torch.float16,
    device_map="auto",
#    quantization_config=bnb_config
)

# set the model to evaluation mode
model.eval()

# load the processor (tokenizer and image processor)
processor = Qwen2VLProcessor.from_pretrained(model_id)
processor.image_processor.size = {'height': pixel_size, 'width': pixel_size}
processor.image_processor.do_resize = True

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define paths to your data folders
metadata_folder = "./datasplit_new/testing/metadata"           # Replace with your actual path
buyer_images_folder = "./datasplit_new/testing/buyer_images"   # Replace with your actual path
seller_images_folder = "./datasplit_new/testing/seller_images" # Replace with your actual path

# instantiate the dataset
dataset = CustomDataset(
    metadata_folder=metadata_folder,
    buyer_images_folder=buyer_images_folder,
    seller_images_folder=seller_images_folder,
    pixel_size=pixel_size
)


def custom_collate_fn(batch):
    return batch

# create a DataLoader for batching with the custom collate function
batch_size = 1  # Keep it as 1 for memory efficiency
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn
)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------
# 6. Define the Inference Function
# ----------------------------------

def generate_predictions(model, processor, batch, device, max_new_tokens=1024, temperature=1.0, top_p=0.9, do_sample=True):
    generated_texts = []

    for sample in batch:
        # Extract system and user messages
        system = next((item for item in sample if item['role'] == 'system'), None)
        user = next((item for item in sample if item['role'] == 'user'), None)

        # Construct the conversation with correct message structure
        conversation = []

        if system:
            conversation.extend(system['content'])

        if user:
            conversation.extend(user['content'])

        # Append the prompt as a text message
        #conversation.append({"type": "text", "text": prompt})


        #conversation = []

        #if system:
        #    for content in system['content']:
        #        if content['type'] == 'text':
        #            conversation.append(content['text'])

        #if user:
        #    for content in user['content']:
        #        if content['type'] == 'text':
        #            conversation.append(content['text'])
        #        elif content['type'] == 'image':
        #            conversation.append(content['image'])  # PIL Image

        # Define the detailed prompt
        prompt = "Given the seller imager, seller description, buyer image and buyer description, I want you to classify the new sample based on these classes:\n-1: Buyer's opinion\n0:Here seller is at fault\n1:Here buyer agreed with the seller\n"

        # Append the prompt to the conversation
        conversation.append({"type": "text", "text": prompt})

        # Prepare messages list for processor
        messages_list = [
            {
                "role": "user",
                "content": conversation
            }
        ]
        #print(messages_list)

        # Apply chat template
        text = processor.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages_list)

        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            #videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Trim the generated tokens to exclude the input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        #print(f"generatedids: {generated_ids}")
        # Decode the generated tokens
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"outpute_text: {output_text}")
        # Append the generated text
        generated_texts.extend(output_text)
    
    return generated_texts

# ----------------------------------
# 7. Run Inference and Collect Results
# ----------------------------------

results = []
k = 0
for batch in tqdm(dataloader, desc="Running Inference"):
    # Generate predictions
    generated_texts = generate_predictions(model, processor, batch, device)
    #print(batch)
    for i, sample in enumerate(batch):
        #if k > 10:
        #    break
        #k += 1
        # Extract true label from assistant message
        assistant = next((item for item in sample if item['role'] == 'assistant'), None)
        true_label_str = assistant['content'][0]['text'] if assistant and 'content' in assistant and len(assistant['content']) > 0 else ""
        true_label = parse_label(true_label_str)

        # Extract generated response
        generated_response = generated_texts[i]
        predicted_label = parse_label(generated_response)

        print(f"true: {true_label} predicted label: {predicted_label}")
        print(f"generated response: {generated_response}")
        # Store the result
        results.append({
            #"input_sample": sample,
            "generated_response": generated_response,
            "predicted_label": predicted_label,
            "true_label": true_label
        })


# save results to a JSON file
with open(f"./testing/{model_name}_base_testing.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)


# extract true labels and predictions
true_labels = [item["true_label"] for item in results]
predictions = [item["predicted_label"] for item in results]

# filter out samples with unknown labels (-37)
filtered_true = []
filtered_pred = []
for t, p in zip(true_labels, predictions):
    if t != -37 and p != -37:
        filtered_true.append(t)
        filtered_pred.append(p)

# calculate accuracy
accuracy = accuracy_score(filtered_true, filtered_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# calculate precision, recall, and f1-score
precision, recall, f1, _ = precision_recall_fscore_support(
    filtered_true, filtered_pred, average='weighted', zero_division=0
)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")

# classification report
target_names = ["Negative", "Neutral/Fault", "Positive"]
print("\nClassification Report:")
print(classification_report(filtered_true, filtered_pred, target_names=target_names, zero_division=0))

