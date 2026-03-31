import torch
import os
from diffusers import StableDiffusionXLPipeline

def test_lora_sdxl(
    model_id = "SG161222/RealVisXL_V4.0",
    lora_path = "./lora_output/lora_step200.safetensors",  # cambia per ogni test
    output_dir = "./test_output"
    subject_name = "subject"
):
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    pipe.unet.load_attn_procs(lora_path)

    # opzionale: forza LoRA (se troppo debole)
    # pipe.unet.set_attn_processor_scale(1.2)

    prompts = {
        "base": f"a photo of {subject_name}",
        "variation": f"a photo of {subject_name} wearing a red jacket, studio lighting",
        "stress": f"a photo of {subject_name}, fantasy style, cinematic lighting, epic scene"
    }

    negative_prompt = "blurry, low quality, distorted"

    # ===== GENERAZIONE =====
    for name, prompt in prompts.items():
        generator = torch.Generator("cuda").manual_seed(42)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        filename = os.path.join(output_dir, f"{os.path.basename(lora_path)}_{name}.png")
        image.save(filename)
        print(f"Salvata: {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="testa un modello LoRA addestrato su SDXL generando alcune immagini")
    parser.add_argument("--lora_path", type=str, default="/content/lora_output/lora_step200.safetensors", help="Percorso del file LoRA da testare")
    parser.add_argument("--output_dir", type=str, default="/content/lora_output", help="Cartella dove salvare i file di output")
    parser.add_argument("--subject_name", type=str, default="subject", help="Nome del soggetto nelle immagini")
    args = parser.parse_args()

    test_lora_sdxl(
        model_id = "SG161222/RealVisXL_V4.0",
        lora_path = args.lora_path,
        output_dir = args.output_dir,
        subject_name = args.subject_name
    )