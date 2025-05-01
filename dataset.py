import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizerFast
from collections import defaultdict
import random

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform=None, max_length=128, split='train', train_ratio=0.8):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform if transform else self._get_default_transform()
        self.max_length = max_length
        self.split = split
        self.image_to_captions = self._load_captions(captions_file)
        self.all_image_ids = list(self.image_to_captions.keys())
        random.shuffle(self.all_image_ids)

        num_train = int(len(self.all_image_ids) * train_ratio)
        if split == 'train':
            self.image_ids = self.all_image_ids[:num_train]
        elif split == 'val':
             self.image_ids = self.all_image_ids[num_train:num_train+10]
        else:
            self.image_ids = self.all_image_ids

        self.samples = []
        for img_id in self.image_ids:
            captions = self.image_to_captions[img_id]
            for caption in captions:
                 self.samples.append({'image_id': img_id, 'caption': caption})


    def _load_captions(self, captions_file):
            image_to_captions = defaultdict(list)
            try:
                with open(captions_file, 'r', encoding='utf-8') as f:
                    header = next(f) 
                    if 'image' not in header or 'caption' not in header:
                        print(f"Warning: Caption file header might be incorrect: {header.strip()}")

                    for line in f:
                        line = line.strip()
                        if not line: continue

                        parts = line.split(',', 1) 
                        if len(parts) == 2:
                            image_name, caption = parts
                            image_name = image_name.strip()
                            caption = caption.strip()
                            if image_name and caption:
                                image_to_captions[image_name].append(caption)

            except FileNotFoundError:
                print(f"Error: Captions file not found at {captions_file}")
                return image_to_captions
            except Exception as e:
                print(f"Error reading or parsing captions file: {e}")
                return image_to_captions 

            if not image_to_captions:
                print("Warning: No captions were loaded. Check file path, format, and content.")

            return image_to_captions

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
         return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['image_id']
        caption = sample['caption']

        img_path = os.path.join(self.image_dir, img_id)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}. Returning None.")
            return None


        encoding = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)


        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption, 
            'image_id': img_id
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]


    return {
        'pixel_values': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(), 
        'captions': captions,
        'image_ids': image_ids
    }