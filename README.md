# my-sd-scripts
using Goolge Colab:  
[Colab](https://colab.research.google.com/)

## upload training set

### From PC

```python
from google.colab import files
import os, shutil

dest_path = '/content/training_data/upload'
os.makedirs(dest_path, exist_ok=True)
uploaded = files.upload()  # Si aprirà un popup per selezionare i file
for filename in uploaded.keys():
    shutil.move(filename, os.path.join(dest_path, filename))
print("Caricamento completato!")
```

### From Google Drive
```python
from google.colab import drive
import os,shutil

drive.mount('/content/drive')
source_path = '/content/drive/MyDrive/lora/dataset/sarmod2'

dest_path = '/content/training_data/upload-drive'
os.makedirs(dest_path, exist_ok=True)

shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
print("Copia completata!")
```

## Data Preparation

```
!pip install -q diffusers transformers accelerate peft safetensors Pillow
!pip install -q bitsandbytes --upgrade
!git clone https://github.com/erebox/my-sd-scripts.git
%cd my-sd-scripts
!python 01-prep.py
```
## Training

!python 02-training.py


## Download lora result

## to PC

```python
import shutil

folder_path = '/content/training_data/upload'
zip_path = '/content/training_data/upload.zip'

shutil.make_archive('/content/training_data/upload', 'zip', folder_path)
files.download(zip_path)
```

## to Google drive
```python
from google.colab import drive
import os, shutil

drive.mount('/content/drive')
source_path = '/content/lora_output'
dest_path = '/content/drive/MyDrive/lora/model/sarmod2'

os.makedirs(dest_path, exist_ok=True)
shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
print("Copia su Drive completata!")
```












import os
import glob
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch

LORA_DIR = "/content/lora_output"  # cartella con checkpoint
PROMPTS = [
    "a photo of sarmod1 in a fantasy setting, cinematic lighting",
    "sarmod1 portrait, soft lighting",
    "sarmod1 in a forest, realistic style",
]

lora_checkpoints = sorted(glob.glob(os.path.join(LORA_DIR, "*.safetensors")))
if not lora_checkpoints:
    raise ValueError("Nessun checkpoint trovato in " + LORA_DIR)
LORA_PATH = lora_checkpoints[-1]
print("Ultimo checkpoint trovato:", LORA_PATH)

pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

lora_weights = load_file(LORA_PATH)
unet_state = pipe.unet.state_dict()
for k, v in lora_weights.items():
    if k in unet_state:
        unet_state[k] = v
pipe.unet.load_state_dict(unet_state)

os.makedirs("test_outputs", exist_ok=True)
for i, prompt in enumerate(PROMPTS):
    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
    path = f"test_outputs/test_{i+1}.png"
    image.save(path)
    print(f"✅ Salvata immagine: {path}")
    display(image)  # su Colab mostra l'immagine