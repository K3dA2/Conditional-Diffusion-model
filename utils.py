import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import uuid

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

def get_data(path):
    image_extensions = ['.jpg','.png']
    image_names = []
    for filename in os.listdir(path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_names.append(filename)
    return image_names


# create a fixed beta schedule
def linear(timesteps=1000):
    beta = np.linspace(0.0001, 0.02, timesteps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha, 0)
    alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)
    return sqrt_alpha_bar, one_minus_sqrt_alpha_bar

def cosine(t, timesteps=1000):
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    # Ensure t is a float tensor
    t = t.float()
    
    # Normalize t by the number of timesteps
    #t = t / timesteps

    # Compute start and end angles using arccos, ensuring the computation stays within PyTorch
    start_angle = torch.arccos(torch.tensor([max_signal_rate], device=t.device))
    end_angle = torch.arccos(torch.tensor([min_signal_rate], device=t.device))
    
    # Calculate diffusion angles
    diffusion_angles = start_angle + t * (end_angle - start_angle)
    
    # Calculate signal and noise rates using PyTorch operations
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates
    
# this function will help us set the RNG key for Numpy
def set_key(key):
    np.random.seed(key)

# this function will add noise to the input as per the given timestamp
def forward_linear_noise(key, x_0, t):
    set_key(key)
    noise = np.random.normal(size=x_0.shape)
    sqrt_alpha_bar, one_minus_sqrt_alpha_bar = linear()
    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t  * ((x_0 - 127.5)/127.5) + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise

def forward_cosine_noise(key, x_0, t):
    set_key(key)
    noise = np.random.normal(size=x_0.shape)
    noise_rates,signal_rates = cosine(t)
    reshaped_noise_rates = np.reshape(noise_rates, (-1, 1, 1, 1))
    reshaped_signal_rates = np.reshape(signal_rates, (-1, 1, 1, 1))
    noisy_image = reshaped_signal_rates  * ((x_0 - 127.5)/127.5) + reshaped_noise_rates * noise
    return noisy_image, noise

def forward_cosine_noise(key, x_0, t,device = device):
    set_key(key)
    # Ensure x_0 is a tensor and on the correct device before operations
    x_0 = x_0.to(device)

    # Generate noise using PyTorch instead of NumPy to keep everything on the same device
    noise = torch.randn_like(x_0)

    # Assuming cosine returns tensors; also, make sure it handles tensors and is device-aware
    noise_rates, signal_rates = cosine(t)

    # Use PyTorch's view for reshaping
    reshaped_noise_rates = noise_rates.view(-1, 1, 1, 1)
    reshaped_signal_rates = signal_rates.view(-1, 1, 1, 1)

    # Element-wise operations in PyTorch
    noisy_image = reshaped_signal_rates * x_0 + reshaped_noise_rates * noise
    return noisy_image, noise


# this function will be used to create sample timestamps between 0 & T
def generate_timestamp(key, num, timesteps=1000):
    set_key(key)
    return torch.randint(0, timesteps,(num,), dtype=torch.int32)

def reshape_img(img,size = (64,64),greyscale = False):
    data = cv2.resize(img,size)
    
    if not greyscale:
        data = np.transpose(data,(2,0,1))
    return data


def forward_noise_test():
    # Let us visualize the output image at a few timestamps
    sample_data = plt.imread("120b40c149fcd761cfc4f5ef4225c9aa.jpg")

    for index, i in enumerate([0,10, 50, 100, 199]):
        noisy_im, noise = forward_cosine_noise(0, np.expand_dims(sample_data, 0), torch.from_numpy(np.array([i,])))
        plt.subplot(1, 5, index+1)
        plt.imshow(np.squeeze(noisy_im,0))
        
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reverse_diffusion(model, diffusion_steps, device=device, show = False, size = (32,32)):
    step_size = 1.0 / diffusion_steps
    current_images = torch.randn(1, 3, size[0], size[1]).to(device)
    model.eval()
    with torch.no_grad():
        for step in range(diffusion_steps):
            diffusion_times = torch.ones((1, 1)).to(device) - step * step_size 
            diffusion_times.to(device)

            # Ensure model and other  operations are also moved to the device
            pred_noises = model(current_images, diffusion_times)
            noise_rates, signal_rates = cosine((diffusion_times.to("cpu")))
            #save_img(pred_noises,'Noise/',t = diffusion_times)

            pred_noises = pred_noises.to(device)  # Move to the specified device
            noise_rates = noise_rates.to(device)  # Move to the specified device
            signal_rates = signal_rates.to(device)  # Move to the specified device
            
            pred_images = (current_images - noise_rates * pred_noises) / signal_rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = cosine(next_diffusion_times.to("cpu"))
            
            next_noise_rates = next_noise_rates.to(device)  # Move to the specified device
            next_signal_rates = next_signal_rates.to(device)  # Move to the specified device
            
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    model.train()
    pred_images = (pred_images.clamp(-1, 1) + 1) / 2

    if show == False:
            plt.imshow(np.transpose(pred_images[-1].cpu().numpy(), (1, 2, 0)))
            plt.axis('off')  # If you want to hide the axes
            # Generate a random filename
            random_filename = str(uuid.uuid4()) + '.png'

            # Specify the directory where you want to save the image
            save_directory = 'Inference Images/'

            # Create the full path including the directory and filename
            full_path = os.path.join(save_directory, random_filename)
            # Save the image with the random filename
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.imshow(np.transpose(pred_images[-1].cpu().numpy(), (1, 2, 0)))
        plt.show()

def reverse_diffusion_cfg(model, diffusion_steps, category, cfg_scale, device=device, show = False, size = (32,32)):
    step_size = 1.0 / diffusion_steps
    current_images = torch.randn(1, 3, size[0], size[1]).to(device)
    model.eval()
    with torch.no_grad():
        for step in range(diffusion_steps):
            diffusion_times = torch.ones((1, 1)).to(device) - step * step_size 
            diffusion_times.to(device)
            category = category.to(device)

            pred_noises = model(current_images, diffusion_times, category)
            # Ensure model and other  operations are also moved to the device
            if cfg_scale > 0:
                uncoditional_pred_noises = model(current_images, diffusion_times)
                pred_noises = pred_noises.to("cpu")
                uncoditional_pred_noises = uncoditional_pred_noises.to("cpu")
                pred_noises = torch.lerp(uncoditional_pred_noises,pred_noises,cfg_scale)
                pred_noises = pred_noises.to(device)

            noise_rates, signal_rates = cosine((diffusion_times.to("cpu")))
            #save_img(pred_noises,'Noise/',t = diffusion_times)

            pred_noises = pred_noises.to(device)  # Move to the specified device
            noise_rates = noise_rates.to(device)  # Move to the specified device
            signal_rates = signal_rates.to(device)  # Move to the specified device
            
            pred_images = (current_images - noise_rates * pred_noises) / signal_rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = cosine(next_diffusion_times.to("cpu"))
            
            next_noise_rates = next_noise_rates.to(device)  # Move to the specified device
            next_signal_rates = next_signal_rates.to(device)  # Move to the specified device
            
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    model.train()
    pred_images = (pred_images.clamp(-1, 1) + 1) / 2

    if show == False:
            plt.imshow(np.transpose(pred_images[-1].cpu().numpy(), (1, 2, 0)))
            plt.axis('off')  # If you want to hide the axes
            # Generate a random filename
            random_filename = str(uuid.uuid4()) + '.png'

            # Specify the directory where you want to save the image
            save_directory = 'Inference Images/'

            # Create the full path including the directory and filename
            full_path = os.path.join(save_directory, random_filename)
            # Save the image with the random filename
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.imshow(np.transpose(pred_images[-1].cpu().numpy(), (1, 2, 0)))
        plt.show()


def save_img(img,path,t,timesteps=1000):
    if torch.is_tensor(img):
        img = img.cpu().numpy()  # Ensure the tensor is moved to CPU and converted to numpy
    img = np.transpose(img[-1], (1, 2, 0))  # Adjust depending on your data format if needed
    
    plt.imshow(img)
    plt.axis('off')  # If you want to hide the axes
    # Generate a random filename
    random_filename = str(uuid.uuid4())+ str(t*timesteps) + '.png'

    # Specify the directory where you want to save the image
    save_directory = path

    # Create the full path including the directory and filename
    full_path = os.path.join(save_directory, random_filename)
    # Save the image with the random filename
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
