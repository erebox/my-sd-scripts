import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LoraDataset(Dataset):
    def __init__(self, folder, tokenizer1, tokenizer2, size=512, default_caption="a photo"):
        self.images = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png"))
        self.captions = []
        for img_path in self.images:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path) as f:
                    self.captions.append(f.read().strip())
            else:
                self.captions.append(default_caption)
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(f"Dataset: {len(self.images)} immagini")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        pixel_values = self.transform(img)
        caption = self.captions[idx]
        tok1 = self.tokenizer1(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        tok2 = self.tokenizer2(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids_1": tok1.input_ids.squeeze(),
            "input_ids_2": tok2.input_ids.squeeze(),
        }

class LoraDatasetV2(Dataset):
    def __init__(self, folder, tokenizer1, tokenizer2, size=512, default_caption="a photo"):
        self.images = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png"))
        self.captions = []
        for img_path in self.images:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path) as f:
                    self.captions.append(f.read().strip())
            else:
                self.captions.append(default_caption)
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        print(f"Dataset: {len(self.images)} immagini")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        pixel_values = self.transform(img)
        caption = self.captions[idx]
        tok1 = self.tokenizer1(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        tok2 = self.tokenizer2(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids_1": tok1.input_ids.squeeze(),
            "input_ids_2": tok2.input_ids.squeeze(),
        }
