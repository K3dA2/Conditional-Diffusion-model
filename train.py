import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,Dataset
from PIL import Image
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from model import UnetConditional
from utils import forward_cosine_noise, reverse_diffusion_cfg, count_parameters,reverse_diffusion
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

def get_data_loader(root_dir, batch_size, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust these values if necessary
    ])
    
    dataset = CustomDataset(root_dir=root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return data_loader


def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, max_grad_norm=1.0, timesteps=200, epoch_start=0, accumulation_steps=4):
    model.train()
    for epoch in range(epoch_start, n_epochs + epoch_start):
        loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        optimizer.zero_grad()  # Initialize the gradient

        for batch_idx, (imgs, labels) in enumerate(progress_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Generate timestamps
            t = torch.randint(0, timesteps, (imgs.size(0),), dtype=torch.float32).to(device) / timesteps
            t = t.view(-1, 1)

            imgs, noise = forward_cosine_noise(None, imgs, t, device= device)

            if np.random.random() <= 0.1:
                outputs = model(imgs, t)
            else:
                outputs = model(imgs, t, labels)

            loss = loss_fn(outputs, noise)

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename

        with open("cifar-diffusion-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        # Save model checkpoint every 20 epochs
        if epoch % 20 == 0:
            model_filename = f'cifar-diffusion-cts_epoch_cfg.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        # Optional: Generate samples every 5 epochs
        if epoch % 5 == 0:
            reverse_diffusion_cfg(model, 30, torch.tensor([[5]], dtype=torch.int32), 5, device = device,size=(32, 32), show = True)

if __name__ == '__main__':
    timesteps = 1000

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    model = UnetConditional()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss().to(device)
    print("param count: ", count_parameters(model))

    reverse_diffusion_cfg(model, 30, torch.tensor([[8]], dtype=torch.int32), 5, device = device,size=(32, 32), show = True)

    root_dir = '/Users/ayanfe/Downloads/archive-2/trainingSet/trainingSet'  # Replace with the path to your dataset
    batch_size = 64
    dataloader = get_data_loader(root_dir, batch_size)

    training_loop(
    n_epochs=1000,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    device=device,
    data_loader=dataloader,
    timesteps=timesteps,
    epoch_start=0,
    accumulation_steps= 1  # Adjust this value as needed
    )


