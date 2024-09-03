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
from model import UnetConditional,Config,Unet
from utils import forward_cosine_noise, reverse_diffusion_cfg, count_parameters,reverse_diffusion
import random
import matplotlib.pyplot as plt


def show_images_and_noise(images, noise, num_images=5):
    images = images[:num_images].cpu().detach().numpy().transpose(0, 2, 3, 1)
    noise = noise[:num_images].cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        axes[0, i].imshow((images[i] * 0.5) + 0.5)  # Denormalize to [0, 1]
        axes[0, i].axis('off')
        axes[0, i].set_title("Original Image")
        
        axes[1, i].imshow((noise[i] * 0.5) + 0.5)  # Denormalize to [0, 1]
        axes[1, i].axis('off')
        axes[1, i].set_title("Noise")

    plt.show()

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label as it's not used

def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.7002, 0.6099, 0.6036), (0.2195, 0.2234, 0.2097))  # Adjust these values if you have RGB images
    ])
    
    # Get the list of all image files in the root directory, excluding non-image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        raise ValueError("No valid image files found in the specified directory.")

    # If num_samples is not specified, use the entire dataset
    if num_samples is None or num_samples > len(image_files):
        num_samples = len(image_files)
    elif num_samples <= 0:
        raise ValueError("num_samples should be a positive integer.")

    print("data length: ", len(image_files))
    
    # Generate a list of indices to sample from (ensure dataset size is not exceeded)
    if shuffle:
        indices = random.sample(range(len(image_files)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Create the subset dataset
    subset_dataset = CustomDataset([image_files[i] for i in indices], transform=transform)
    
    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader


def validation_loop(model, loss_fn, device, val_loader):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate timestamps
            t = torch.randint(0, timesteps, (imgs.size(0),), dtype=torch.float32).to(device) / timesteps
            t = t.view(-1, 1)

            imgs, noise = forward_cosine_noise(None, imgs, t, device=device)
            outputs = model(imgs, t)
            loss = loss_fn(outputs, noise)

            val_loss += loss.item()

    return val_loss / len(val_loader)

def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, val_loader,max_grad_norm=1.0, timesteps=1000, epoch_start=0, accumulation_steps=4):
    for epoch in range(epoch_start, n_epochs + epoch_start):
        model.train()
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


            outputs = model(imgs, t)

            loss = loss_fn(outputs, noise)

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename

        with open("waifu-diffusion-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        # Save model checkpoint every 20 epochs
        if epoch % 1 == 0:
            model_filename = f'64w-waifu-diffusion.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        # Optional: Generate samples every 100 epochs
        if epoch % 100 == 0:
            with torch.no_grad():
                reverse_diffusion(model,50, size=(64,64))
                

if __name__ == '__main__':
    timesteps = 1000

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    config = Config(width=64)
    model = Unet(config)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss().to(device)
    print("param count: ", count_parameters(model))

    
    path = ''
    val_path = ''
    model_path = ''
    epoch = 0

    dataloader = get_data_loader(path, batch_size=8, num_samples=20_000)
    val_loader = get_data_loader(val_path,batch_size=8)
    
    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    '''
    
    with torch.no_grad():
        reverse_diffusion(model,50, size=(64,64))

    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=dataloader,
        val_loader= val_loader,
        timesteps=timesteps,
        epoch_start = epoch + 1,
        accumulation_steps= 1  # Adjust this value as needed
    )
    
    
    


