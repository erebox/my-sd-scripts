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
from dataset import LoraDatasetV2 as LoraDataset

def train_lora_sdxl(
    model_id,
    dataset_path,
    output_path,
    steps=1000,
    lr=1e-4,
    batch_size=1,
    save_every=100,
    resolution=512,
    device="cuda",
    dtype=torch.bfloat16,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    default_caption="a photo of subject",
    warmup_steps=50,
    early_stop_patience=200
):
    os.makedirs(output_path, exist_ok=True)

    print(f"PyTorch: {torch.__version__}, CUDA disponibile: {torch.cuda.is_available()}")
    if device=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Caricamento modello...")
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True).to(device)
    unet, vae = pipe.unet, pipe.vae
    tokenizer_1, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    noise_scheduler = pipe.scheduler

    print("Configurazione LoRA...")
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                             target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=lora_dropout)
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = LoraDataset(dataset_path, tokenizer_1, tokenizer_2, size=resolution, default_caption=default_caption)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)

    unet.train()
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    global_step = 0
    data_iter = iter(dataloader)
    best_loss = float("inf")
    steps_since_best = 0
    rolling_loss = []

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
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            enc1_out = pipe.text_encoder(input_ids_1, output_hidden_states=True)
            enc2_out = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
            text_embeds = torch.cat([enc1_out.hidden_states[-2], enc2_out.hidden_states[-2]], dim=-1)
            pooled_embeds = enc2_out.last_hidden_state.mean(dim=1)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        added_cond_kwargs = {"text_embeds": pooled_embeds}

        noise_pred = unet(noisy_latents, timesteps, text_embeds, added_cond_kwargs=added_cond_kwargs).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        rolling_loss.append(loss.item())
        if len(rolling_loss) > 10: rolling_loss.pop(0)
        avg_loss = sum(rolling_loss)/len(rolling_loss)

        global_step += 1

        if global_step % 10 == 0:
            print(f"Step {global_step}/{steps} | loss rolling media: {avg_loss:.4f}")

        # Checkpointing
        if global_step % save_every == 0:
            path = f"{output_path}/lora_step{global_step}.safetensors"
            lora_weights = {k:v for k,v in unet.state_dict().items() if "lora" in k}
            save_file(lora_weights, path)
            print(f"Checkpoint salvato: {path}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            steps_since_best = 0
        else:
            steps_since_best += 1
            if steps_since_best >= early_stop_patience:
                print(f"Early stopping attivato dopo {global_step} step")
                break

    # Salvataggio finale
    final_path = f"{output_path}/lora_final.safetensors"
    lora_weights = {k:v for k,v in unet.state_dict().items() if "lora" in k}
    save_file(lora_weights, final_path)
    print(f"Training completato! File finale salvato: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Addestra LoRA su SDXL con early stopping")
    parser.add_argument("--dataset_path", type=str, default="./training_data")
    parser.add_argument("--output_path", type=str, default="./lora_output")
    args = parser.parse_args()

    train_lora_sdxl(
        model_id="SG161222/RealVisXL_V4.0",
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        steps=1000,
        lr=1e-4,
        batch_size=1,
        resolution=512,
        default_caption="a photo of subject",
        early_stop_patience=150
    )