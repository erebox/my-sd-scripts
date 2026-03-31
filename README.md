# my-sd-scripts
using Goolge Colab:  
[Colab](https://colab.research.google.com/)

## upload training set

### From PC

    from google.colab import files
    import os, shutil

    dest_path = '/content/training_data/upload-pc'
    os.makedirs(dest_path, exist_ok=True)
    uploaded = files.upload()  # Si aprirà un popup per selezionare i file
    for filename in uploaded.keys():
        shutil.move(filename, os.path.join(dest_path, filename))
    print("Caricamento completato!")

### From Google Drive

    from google.colab import drive
    import shutil
    import os

    drive.mount('/content/drive')
    source_path = '/content/drive/MyDrive/lora/dataset/upload'

    dest_path = '/content/training_data/upload-drive'
    os.makedirs(dest_path, exist_ok=True)

    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
    print("Copia completata!")


## Data Preparation

    !pip install -q diffusers transformers accelerate peft safetensors Pillow
    !pip install -q bitsandbytes --upgrade
    !git clone https://github.com/erebox/my-sd-scripts.git
    %cd my-sd-scripts
    !python 01-prep.py

## Training

    !python 02-training.py
