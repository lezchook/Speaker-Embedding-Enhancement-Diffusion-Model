import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings
    
class ResidualFCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_embed_dim):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        self.silu = nn.SiLU()
        
    def forward(self, x, t_emb):
        residual = x
        
        x = self.norm1(x)
        x = self.silu(x)
        x = self.fc1(x)
        
        x = x + self.time_proj(t_emb)

        x = self.norm2(x)
        x = self.silu(x)
        x = self.fc2(x)
        
        return x + residual

class SEED(nn.Module):
    def __init__(self, embedding_dim, num_blocks=3, time_embed_dim=None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 2
        
        if time_embed_dim is None:
            time_embed_dim = embedding_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = SinusoidalPositionEmbedding(time_embed_dim)

        self.blocks = nn.ModuleList([
            ResidualFCBlock(embedding_dim, self.hidden_dim, time_embed_dim)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        
        x = x_t
        for block in self.blocks:
            x = block(x, t_emb)
        
        return x

class DiffusionProcess:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):
        self.T = T
        
        self.betas = self._get_scaled_linear_schedule(T, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _get_scaled_linear_schedule(self, T, beta_start, beta_end):
        scale = 1000.0 / T
        beta_start_scaled = beta_start * scale
        beta_end_scaled = beta_end * scale
        return torch.linspace(beta_start_scaled, beta_end_scaled, T, dtype=torch.float32)
    
    def forward_diffusion(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)
        
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
    
def train_seed_epoch(seed_model, diffusion, dataloader, optimizer, scheduler, device):
    seed_model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training SEED", leave=True, dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress_bar):
        x_clean = batch["clean"].to(device).squeeze(0)  # (D,)
        y_noisy = batch["noisy"].to(device).squeeze(0)  # (K, D)
        num_augmentations = y_noisy.size(0)

        t = torch.randint(0, diffusion.T, (1,), device=device)  # (1,)

        # clean forward
        noise_clean = torch.randn_like(x_clean)
        x_t, _ = diffusion.forward_diffusion(x_clean, t, noise=noise_clean)
        x_pred = seed_model(x_t, t).squeeze(0)
        loss_clean = F.mse_loss(x_pred, x_clean)

        # noisy forward
        loss_noisy = 0
        for k in range(num_augmentations):
            y_k = y_noisy[k]
            noise_k = torch.randn_like(y_k)
            y_t, _ = diffusion.forward_diffusion(y_k, t, noise=noise_k)
            y_pred = seed_model(y_t, t).squeeze(0)
            loss_noisy += F.mse_loss(y_pred, x_clean)

        loss_noisy /= num_augmentations
        loss = loss_clean + loss_noisy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seed_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})

    scheduler.step()
    avg_loss = total_loss / num_batches
    progress_bar.close()

    return avg_loss

def saveParameters(model, optimizer, scheduler, num_epoch, best_loss, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "num_epoch": num_epoch,
        "best_loss": best_loss
    }
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = f"seed_model_{str(num_epoch).zfill(4)}.pth"
    save_path = os.path.join(path, filename)
    torch.save(checkpoint, save_path)
    
    return save_path

def loadParameters(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    num_epoch = checkpoint["num_epoch"]

    print(f"Loaded checkpoint from epoch {num_epoch}")
    
    return num_epoch, checkpoint["best_loss"]

@torch.no_grad()
def validate_seed(seed_model, val_loader, device, t=50):
    seed_model.eval()
    total_cos_sim = 0.0
    num_samples = 0

    for batch in val_loader:
        x_clean = batch["clean"].to(device)
        y_noisy = batch["noisy"].to(device)
        
        batch_size = x_clean.size(0)
        num_augmentations = y_noisy.size(1)
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        for k in range(num_augmentations):
            y_k = y_noisy[:, k, :]
            y_pred = seed_model(y_k, t_tensor)
            cos_sim = F.cosine_similarity(y_pred, x_clean, dim=-1)
            total_cos_sim += cos_sim.sum().item()
            num_samples += batch_size

    avg_cos_sim = total_cos_sim / num_samples
    return avg_cos_sim

def train_seed(seed_model, diffusion, train_loader, val_loader, optimizer,
               scheduler, num_epochs=60, start_epoch=0, best_loss=None, 
               device="cuda", checkpoint_dir="./checkpoints", 
               best_model_dir="./best_models"):
    
    os.makedirs(checkpoint_dir, exist_ok=True) 
    os.makedirs(best_model_dir, exist_ok=True)

    seed_model = seed_model.to(device)
    diffusion = diffusion.to(device)
    
    print("=" * 60)
    print(f"Starting SEED training from {start_epoch} -> {num_epochs-1} epoch.")
    print(f"Embedding dimension: {seed_model.embedding_dim}")
    print(f"Device: {device}")
    print("=" * 60)

    if best_loss is None:
        best_loss = float("inf")
    
    for epoch in range(start_epoch, num_epochs):
        print("=" * 60)
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("=" * 60)
        
        train_loss = train_seed_epoch(
            seed_model=seed_model, diffusion=diffusion, dataloader=train_loader,
            optimizer=optimizer, scheduler=scheduler, device=device
        )
        
        print(f"Epoch {epoch:03d}, Loss (train set) {train_loss:.6f}")

        checkpoint_path = saveParameters(
            model=seed_model, optimizer=optimizer, scheduler=scheduler, 
            num_epoch=epoch, best_loss=best_loss, path=checkpoint_dir
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = saveParameters(
                seed_model, optimizer, scheduler, 
                epoch, train_loss, best_model_dir
            )
            print(f"New best model with Loss: {best_loss:.6f}")
            print(f"Saved to: {best_model_path}")

        if val_loader is not None:
            val_cos_sim = validate_seed(seed_model=seed_model, val_loader=val_loader, device=device)
            print(f"Epoch {epoch:03d}, Validation Cos_Sim: {val_cos_sim:.4f}")

    print("=" * 60)
    print("Training completed!")
    print("=" * 60)

@torch.no_grad()
def inference_seed(seed_model, embeddings, t=50, device="cuda", use_ensemble=False):
    seed_model.eval()
    
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    embeddings = embeddings.to(device)
    batch_size = embeddings.size(0)
    
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

    enhanced = seed_model(embeddings, t_tensor)
    
    if use_ensemble:
        enhanced = embeddings + enhanced
    
    if squeeze_output:
        enhanced = enhanced.squeeze(0)
    
    return enhanced