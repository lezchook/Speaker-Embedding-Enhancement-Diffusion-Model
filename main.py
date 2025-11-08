import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BasicBlock, ResNet, MainModel
from dataset import test_dataset_loader, SEEDEmbeddingDataset
from LossFunction import AAMSoftmaxLoss
from diffusion import SEED, DiffusionProcess, train_seed, loadParameters
from common import download_dataset, concatenate, extract_dataset, part_extract, download_protocol, split_musan


class Config:
    def __init__(self, config_dict):
        # Paths
        data = config_dict.get('data', {})
        self.data_dir = Path(data.get('data_dir', './data'))
        self.train_path = data.get('train_path', str(self.data_dir / "voxceleb1_dev/wav"))
        self.val_path = data.get('val_path', str(self.data_dir / "voxceleb1_dev/wav"))
        self.test_path = data.get('test_path', str(self.data_dir / "voxceleb1_test/wav"))
        self.musan_path = data.get('musan_path', str(self.data_dir / "musan_split"))
        self.rir_path = data.get('rir_path', str(self.data_dir / "RIRS_NOISES/simulated_rirs"))
        
        # Models
        models = config_dict.get('models', {})
        self.checkpoint_dir = Path(models.get('checkpoint_dir', './checkpoints'))
        self.best_model_dir = Path(models.get('best_model_dir', './best_models'))
        self.pretrained_model = models.get('pretrained_model', './data/models/lab3_model_0039.pth')
        self.resume_checkpoint = models.get('resume_checkpoint', None)
        
        # Training params
        training = config_dict.get('training', {})
        self.batch_size_train = training.get('batch_size_train', 128)
        self.batch_size_val = training.get('batch_size_val', 128)
        self.batch_size_test = training.get('batch_size_test', 128)
        self.num_epochs = training.get('num_epochs', 60)
        self.learning_rate = training.get('learning_rate', 0.0005)
        self.num_workers = training.get('num_workers', 4)
        
        scheduler = training.get('scheduler', {})
        self.scheduler_step = scheduler.get('step_size', 20)
        self.scheduler_gamma = scheduler.get('gamma', 0.5)
        
        # Architecture
        arch = config_dict.get('architecture', {})
        self.n_mels = arch.get('n_mels', 40)
        self.log_input = arch.get('log_input', True)
        self.layers = arch.get('layers', [3, 4, 6, 3])
        self.num_filters = arch.get('num_filters', [32, 64, 128, 256])
        self.embedding_dim = arch.get('embedding_dim', 512)
        
        # Loss
        loss = config_dict.get('loss', {})
        self.margin = loss.get('margin', 0.35)
        self.scale = loss.get('scale', 32.0)
        
        # Diffusion
        diff = config_dict.get('diffusion', {})
        self.num_blocks = diff.get('num_blocks', 3)
        self.diffusion_steps = diff.get('steps', 1000)
        self.beta_start = diff.get('beta_start', 0.0001)
        self.beta_end = diff.get('beta_end', 0.02)
        
        # Data params
        data_params = config_dict.get('data_params', {})
        self.max_frames_train = data_params.get('max_frames_train', 200)
        self.max_frames_val = data_params.get('max_frames_val', 1000)
        self.max_frames_test = data_params.get('max_frames_test', 1000)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(yaml.safe_load(f))


def download_data(data_dir):
    lists_dir = data_dir / "lists"
    
    with open(lists_dir / "datasets.txt") as f:
        download_dataset(f.readlines(), user="***", password="***", save_path=str(data_dir))
    
    with open(lists_dir / "concat_arch.txt") as f:
        concatenate(f.readlines(), save_path=str(data_dir))
    
    extract_dataset(str(data_dir / "voxceleb1_test"), str(data_dir / "vox1_test_wav.zip"))
    extract_dataset(str(data_dir / "voxceleb1_dev"), str(data_dir / "vox1_dev_wav.zip"))
    
    with open(lists_dir / "protocols.txt") as f:
        download_protocol(f.readlines(), save_path=str(data_dir / "voxceleb1_test"))
    
    with open(lists_dir / "augment_datasets.txt") as f:
        download_dataset(f.readlines(), user=None, password=None, save_path=str(data_dir))
    
    extract_dataset(str(data_dir), str(data_dir / "musan.tar.gz"))
    part_extract(str(data_dir), str(data_dir / "rirs_noises.zip"),
                 target=["RIRS_NOISES/simulated_rirs/mediumroom", "RIRS_NOISES/simulated_rirs/smallroom"])
    split_musan(str(data_dir))


def prepare_data_lists(data_dir):
    train_list, val_list, test_list = [], [], []
    black_list = set(os.listdir(data_dir / "voxceleb1_test/wav"))
    num_train_spk = set()
    
    with open(data_dir / "voxceleb1_test/iden_split.txt") as f:
        for line in f:
            split_id, file_path = line.strip().split(' ')
            spk_id = file_path.split('/')[0]
            
            if spk_id in black_list:
                continue
            
            num_train_spk.add(spk_id)
            entry = f"{spk_id} {file_path}"
            
            if split_id == '1':
                train_list.append(entry)
            elif split_id == '2':
                val_list.append(entry)
            elif split_id == '3':
                test_list.append(entry)
    
    return train_list, val_list, test_list, len(num_train_spk)


def build_models(config, num_speakers):
    # Speaker model
    basic_model = ResNet(BasicBlock, layers=config.layers, activation=nn.ReLU,
                        num_filters=config.num_filters, nOut=config.embedding_dim,
                        n_mels=config.n_mels, log_input=config.log_input)
    
    loss_fn = AAMSoftmaxLoss(nOut=config.embedding_dim, nClasses=num_speakers,
                            margin=config.margin, scale=config.scale)
    
    speaker_model = MainModel(basic_model, loss_fn).to(config.device)
    
    if config.pretrained_model and os.path.exists(config.pretrained_model):
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)
        speaker_model.load_state_dict(checkpoint['model'])
    
    # SEED model
    seed_model = SEED(embedding_dim=config.embedding_dim, num_blocks=config.num_blocks,
                    time_embed_dim=config.embedding_dim).to(config.device)
    
    diffusion = DiffusionProcess(T=config.diffusion_steps, 
                                beta_start=config.beta_start,
                                beta_end=config.beta_end)
    
    return speaker_model, seed_model, diffusion


def create_dataloaders(config, speaker_model, train_list, val_list, test_list):
    pin_memory = config.device == 'cuda'
    
    train_dataset = SEEDEmbeddingDataset(speaker_model, train_list, config.train_path,
                                        config.max_frames_train, config.musan_path, config.rir_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_train,
                            shuffle=True, num_workers=config.num_workers, pin_memory=pin_memory)
    
    val_dataset = SEEDEmbeddingDataset(speaker_model, val_list, config.val_path,
                                      config.max_frames_val, config.musan_path, config.rir_path)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val,
                          shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory)
    
    test_dataset = test_dataset_loader(test_list, config.max_frames_test, config.test_path)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers)
    
    return train_loader, val_loader, test_loader


def setup_optimizer(config, seed_model):
    optimizer = torch.optim.AdamW(seed_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,
                                               gamma=config.scheduler_gamma)
    
    start_epoch, best_loss = 0, None
    
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        start_epoch, best_loss = loadParameters(seed_model, optimizer, scheduler, config.resume_checkpoint)
        start_epoch += 1
    
    return optimizer, scheduler, start_epoch, best_loss


def train(config):
    # Prepare data
    train_list, val_list, test_list, num_speakers = prepare_data_lists(config.data_dir)
    
    # Build models
    speaker_model, seed_model, diffusion = build_models(config, num_speakers)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, speaker_model, train_list, val_list, test_list)
    
    # Setup training
    optimizer, scheduler, start_epoch, best_loss = setup_optimizer(config, seed_model)
    
    # Create save directories
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.best_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    train_seed(seed_model, diffusion, train_loader, val_loader, optimizer, scheduler,
              config.num_epochs, start_epoch, best_loss, config.device,
              str(config.checkpoint_dir), str(config.best_model_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--download", default=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    
    config = Config.from_yaml(args.config)
    
    if args.download:
        download_data(config.data_dir)
    
    train(config)


if __name__ == "__main__":
    main()