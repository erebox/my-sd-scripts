import os
import torch
from PIL import Image
import argparse

def prepara_dataset(
    upload_path="/content/training_data/upload",
    dest_path="/content/training_data/soggetto",
    image_size=512
):
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("GPU non disponibile, uso CPU")
    print(f"PyTorch: {torch.__version__}")
    print("Ambiente pronto!")

    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(dest_path, exist_ok=True)

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

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Errore su {nome}: {e}")
            continue

        w, h = img.size
        lato = min(w, h)
        img = img.crop(((w - lato)//2, (h - lato)//2, (w + lato)//2, (h + lato)//2))
        img = img.resize((image_size, image_size), Image.LANCZOS)

        base_name = os.path.splitext(nome)[0]
        out_img = os.path.join(dest_path, base_name + ".jpg")
        img.save(out_img, quality=95)

        caption = captions.get(nome.lower()) or captions.get(base_name.lower())

        if caption:
            with open(os.path.join(dest_path, base_name  + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption)
        else:
            print(f"Nessuna caption per: {nome}")
    print(f"{len(foto)} foto pronte con caption")

    return {
        "num_immagini": len(foto),
        "num_caption": len(captions)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepara un dataset di immagini con caption")
    parser.add_argument("--upload_path", type=str, default="/content/training_data/upload", help="Cartella con le immagini originali e captions.txt")
    parser.add_argument("--dest_path", type=str, default="/content/training_data/soggetto", help="Cartella dove salvare le immagini preprocessate")
    parser.add_argument("--image_size", type=int, default=512, help="Dimensione finale delle immagini (quadrato)")
    
    args = parser.parse_args()
    prepara_dataset(args.upload_path, args.dest_path, args.image_size)