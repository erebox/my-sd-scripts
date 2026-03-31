from google.colab import files, drive
import os
import torch
from PIL import Image

def prepara_dataset(
    src_mode="pc",  # "pc" oppure "drive"
    drive_subpath=None,  # es: "lora/dataset/soggetto"
    upload_path="/content/training_data/upload",
    dest_path="/content/training_data/soggetto",
    image_size=512
):
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print("Ambiente pronto!")

    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(dest_path, exist_ok=True)

    if src_mode == "drive":
        if not drive_subpath:
            raise ValueError("Devi specificare drive_subpath se src_mode='drive'")
        drive.mount('/content/drive')
        drive_path = f"/content/drive/MyDrive/{drive_subpath.strip('/')}"
        if not os.path.exists(drive_path):
            raise ValueError("Percorso Google Drive non valido")
        count = 0
        for filename in os.listdir(drive_path):
            src_file = os.path.join(drive_path, filename)
            dst_file = os.path.join(upload_path, filename)
            if os.path.isfile(src_file):
                with open(src_file, "rb") as f_src, open(dst_file, "wb") as f_dst:
                    f_dst.write(f_src.read())
                count += 1
        print(f"Copiati {count} file da Google Drive")
    elif src_mode == "pc":
        print("Carica tutte le foto JPG/PNG e il file captions.txt")
        uploaded = files.upload()

        for filename in uploaded.keys():
            with open(os.path.join(upload_path, filename), "wb") as f:
                f.write(uploaded[filename])

        print(f"Caricati {len(uploaded)} file")
    else:
        raise ValueError("src_mode deve essere 'pc' o 'drive'")

    captions_file = os.path.join(upload_path, "captions.txt")
    if not os.path.exists(captions_file):
        raise FileNotFoundError("captions.txt non trovato")
    captions = {}
    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                nome, caption = line.split("|", 1)
                captions[nome.strip()] = caption.strip()
    print(f"Lette {len(captions)} caption")

    foto = [f for f in os.listdir(upload_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for nome in foto:
        img_path = os.path.join(upload_path, nome)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        lato = min(w, h)
        img = img.crop(((w - lato)//2, (h - lato)//2, (w + lato)//2, (h + lato)//2))
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img.save(os.path.join(dest_path, nome), quality=95)
        base_name = os.path.splitext(nome)[0]
        caption = (captions.get(nome) or captions.get(nome.lower()) or captions.get(base_name) or captions.get(base_name.lower()))
        if caption:
            txt_name = base_name  + ".txt"
            with open(os.path.join(dest_path, txt_name), "w", encoding="utf-8") as f:
                f.write(caption)
        else:
            print(f"Nessuna caption per: {nome}")
    print(f"{len(foto)} foto pronte con caption")

    return {
        "num_immagini": len(foto),
        "num_caption": len(captions)
    }

if __name__ == '__main__':
    prepara_dataset(src_mode="pc")