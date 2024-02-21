import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
from noise_layer import AddNoise
from safetensors.torch import save_file

# Define the denoising neural network with your custom noise layer
class NoiseAdditiveLayer(nn.Module):
    def __init__(self, noise_std):
        super(NoiseAdditiveLayer, self).__init__()
        self.noise_layer = AddNoise(
            threshold=0.06, exponent=0.0, slope=1.0, 
            intercept=7.0, noise_type='uniform', mean=0.1, std=0.1,
            grid_size=(36, 27), circle_size_factor=0.24, 
            heightSkew=0.25, widthSkew=0.4
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_noisy = self.noise_layer(x)
        x_encoded = self.encoder(x_noisy)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


def adjust_noise_parameters(model, epoch):
    # Adjust noise parameters based on the current epoch
    model.noise_layer.std = 0.08*epoch
    # model.noise_layer.mean = 0.1
    model.noise_layer.heightSkew = 0.25
    model.noise_layer.widthSkew = 0.4
    model.noise_layer.grid_size = (int((epoch*2)**2), int(((epoch*2)**2)*0.75))
    # model.noise_layer.circle_size_factor = 1 - (epoch*0.08)


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        # adjust_noise_parameters(model, epoch)
        model.train()
        running_loss = 0.0
        
        for images in train_loader:
            # Extract the noisy images from the batch
            noisy_images_batch = images[0]  # Assuming the noisy images are the first element of each batch
            
            # Apply noise layer to each noisy image in the batch
            noisy_images = [model.noise_layer(image) for image in noisy_images_batch]
            noisy_images = torch.stack(noisy_images)  # Convert list of noisy images to tensor
            # Remove the extra dimension
            noisy_images = noisy_images.squeeze(1)
            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, noisy_images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # torch.save(model.state_dict(), "denoiser_model.pth")
    save_model_as_safetensor(model, "denoiser_model.safetensors")


def save_model_as_safetensor(model, filename):
    model_state_dict = model.state_dict()
    save_file(model_state_dict, filename)


def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            
            # Apply transformations to convert the image to a tensor
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize images to a consistent size
                transforms.ToTensor(),           # Convert images to PyTorch tensors
            ])
            image_tensor = transform(image)
            
            images.append(image_tensor)

    return torch.stack(images)

# Example of how to use the denoising neural network with your custom noise layer
def main():
    # Load your dataset and create clean images
    # directory = "/home/dev/code/data/training_data"
    directory = "/home/dev/code/data/test_data/people"
    clean_images_tensor = load_images_from_directory(directory)
    # model.noise_layer.circle_size_factor = 1 - (epoch*0.08)


    # Initialize the denoising neural network print(type(images))with your custom noise layer
    denoiser = NoiseAdditiveLayer(noise_std=0.2)
    train_dataset = TensorDataset(clean_images_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_model(denoiser, train_loader)

if __name__ == "__main__":
    main()
