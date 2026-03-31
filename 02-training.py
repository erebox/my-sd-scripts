import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
import argparse

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


def train_lora_sdxl(
    model_id,
    dataset_path,
    output_path,
    steps=3000,
    lr=2e-4,
    batch_size=1,
    save_every=500,
    resolution=512,
    device="cuda",
    dtype=torch.bfloat16,
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.0,
    default_caption="a photo of subject",
    warmup_steps=150
):
    os.makedirs(output_path, exist_ok=True)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Caricamento modello...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)
    unet = pipe.unet
    vae = pipe.vae
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = pipe.scheduler

    print("Configurazione LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    dataset = LoraDataset(
        dataset_path,
        tokenizer_1,
        tokenizer_2,
        size=resolution,
        default_caption=default_caption
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps
    )

    print("Training avviato!")
    unet.train()
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    global_step = 0
    data_iter = iter(dataloader)
    while global_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        input_ids_1 = batch["input_ids_1"].to(device)
        input_ids_2 = batch["input_ids_2"].to(device)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            enc1_out = pipe.text_encoder(input_ids_1, output_hidden_states=True)
            enc2_out = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
            text_embeds = torch.cat(
                [enc1_out.hidden_states[-2], enc2_out.hidden_states[-2]],
                dim=-1
            )
            pooled_embeds = enc2_out[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            device=device,
            dtype=dtype
        )
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids
        }
        noise_pred = unet(
            noisy_latents,
            timesteps,
            text_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        if global_step % 50 == 0:
            print(f"Step {global_step}/{steps} | loss: {loss.item():.4f}")

        if global_step % save_every == 0:
            path = f"{output_path}/lora_step{global_step}.safetensors"
            lora_weights = {k: v for k, v in unet.state_dict().items() if "lora" in k}
            save_file(lora_weights, path)
            print(f"Salvato: {path}")
    print("Training completato!")

    final_path = f"{output_path}/lora_final.safetensors"
    lora_weights = {k: v for k, v in unet.state_dict().items() if "lora" in k}
    save_file(lora_weights, final_path)

    print(f"File finale: {final_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepara un dataset di immagini con caption")
    parser.add_argument("--dataset_path", type=str, default="/content/training_data/soggetto", help="Cartella dove salvare le immagini preprocessate")
    parser.add_argument("--output_path", type=str, default="/content/lora_output", help="Cartella dove salvare i file di output")
    
    args = parser.parse_args()
    train_lora_sdxl(
        model_id="SG161222/RealVisXL_V4.0",
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        steps=3000, lr=0.0002, batch_size=1, resolution=512,
        default_caption="a photo of model"
    )