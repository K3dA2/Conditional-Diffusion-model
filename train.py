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
from model import UnetConditional,Config
from utils import forward_cosine_noise, reverse_diffusion_cfg, count_parameters,reverse_diffusion
import pandas as pd
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

# Define the list of classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame.iloc[:, 0] = self.data_frame.iloc[:, 0].astype(str)  # Ensure the first column is string type
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        # Try adding common image extensions if the file is not found
        if not os.path.exists(img_name):
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.exists(img_name + ext):
                    img_name = img_name + ext
                    break
            else:
                raise FileNotFoundError(f"Image file '{img_name}' not found with common extensions.")
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]
        label_idx = class_to_idx[label]
        label_one_hot = np.zeros(len(classes), dtype=np.float32)
        label_one_hot[label_idx] = 1.0
        
        if self.transform:
            image = self.transform(image)
        
        # Find the index of the highest value
        max_index = np.argmax(label_one_hot)
        return image, torch.tensor(max_index)


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

            #show_images_and_noise(imgs, noise, num_images=5)

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
        if epoch % 5 == 0:
            model_filename = f'cifar-diffusion-cts_epoch_cfg.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        # Optional: Generate samples every 5 epochs
        if epoch % 150 == 0:
            reverse_diffusion_cfg(model, 30, torch.tensor([[3]], dtype=torch.int32), 7, device = device,size=(32, 32), show = True)
            reverse_diffusion_cfg(model, 30, torch.tensor([[3]], dtype=torch.int32), 0, device = device,size=(32, 32), show = True)

if __name__ == '__main__':
    timesteps = 1000

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    config = Config(width=32)
    model = UnetConditional(config)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss().to(device)
    print("param count: ", count_parameters(model))

    #reverse_diffusion_cfg(model, 30, torch.tensor([[8]], dtype=torch.int32), 5, device = device,size=(32, 32), show = True)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the image to 64x64 pixels
        transforms.ToTensor(),       # Convert the image to a PyTorch tensor
        transforms.Normalize((0.4907, 0.4815, 0.4469),(0.1903, 0.1881, 0.1914))
    ])

    # Create the dataset
    csv_file = '/Users/ayanfe/Documents/Datasets/cifar-10/train/trainLabels.csv'  # Path to the CSV file
    root_dir = '/Users/ayanfe/Documents/Datasets/cifar-10/train'         # Directory containing the images and CSV file
    dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)


    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Optionally load model weights if needed
    model_path = "weights/cifar-diffusion-cts_epoch_cfg.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    reverse_diffusion_cfg(model,30,torch.tensor([[6]],dtype=torch.int32),5,size=(32,32),show=True, device=device)
    reverse_diffusion_cfg(model, 30, torch.tensor([[6]], dtype=torch.int32), 0, device = device,size=(32, 32), show = True)
    
    '''
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=dataloader,
        timesteps=timesteps,
        epoch_start = epoch + 1,
        accumulation_steps= 1  # Adjust this value as needed
    )
    '''
    


