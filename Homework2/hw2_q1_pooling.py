# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST, INFO
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define device (Updated to include Mac M1/M2/M3 support)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    
print(f"Using device: {device}")

# --------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001

# --------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------
data_flag = 'bloodmnist'
print(f"Dataset: {data_flag}")
info = INFO[data_flag]
n_classes = len(info['label'])
print(f"Number of classes: {n_classes}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Datasets
# NOTE: download=True will handle the 'bloodmnist.npz' requirement
train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # 1. Convolution layer: 3 input channels -> 32 output channels
        # Kernel 3x3, Stride 1, Padding 1 (Preserves 28x28 spatial size)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # 2. Convolution layer: 32 -> 64 output channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 3. Convolution layer: 64 -> 128 output channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculation for Linear Layer Input with MaxPool:
        # Input image is 28x28.
        # Block 1: Conv(28x28) -> Pool(2) -> 14x14
        # Block 2: Conv(14x14) -> Pool(2) -> 7x7
        # Block 3: Conv(7x7)   -> Pool(2) -> 3x3 (integer floor of 7/2)
        # Final block output shape: (128 channels, 3 height, 3 width)
        # Flattened size = 128 * 3 * 3 = 1,152
        self.flattened_size = 128 * 3 * 3
        
        # 4. Linear Layer: Flattened Input -> 256 features
        self.fc1 = nn.Linear(self.flattened_size, 256)
        
        # 5. Linear Layer: 256 -> Number of Classes (8)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x) # Added MaxPool
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x) # Added MaxPool
        
        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x) # Added MaxPool
        
        # Flatten
        # x.size(0) is the batch size
        x = x.view(x.size(0), -1) 
        
        # Linear 1
        x = self.fc1(x)
        x = F.relu(x)
        
        # Linear 2 (Output)
        x = self.fc2(x)
        
        # x = F.softmax(x, dim=1) 
        
        return x

# Initialize Model, Loss, and Optimizer
model = SimpleCNN(num_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------------------------------------------
# Training Helper Functions
# --------------------------------------------------------------

def train_epoch(loader, model, criterion, optimizer):
    model.train() # Set model to training mode
    running_loss = 0.0
    
    for imgs, labels in loader:
        # Move data to GPU/CPU
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device) # Labels need to be 1D LongTensor for CrossEntropy
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(imgs)
        
        # 3. Compute loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(loader, model):
    model.eval() # Set model to evaluation mode
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(imgs)
            # Get the class with the highest score
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.cpu().tolist()

    return accuracy_score(targets, preds)

def plot(epochs_range, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs_range, plottable)
    plt.title(f"{ylabel} over Time")
    plt.grid(True)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')
    # plt.show() # Uncomment if running locally with a display

# --------------------------------------------------------------
# Main Training Loop
# --------------------------------------------------------------
print("Starting training...")
total_start = time.time()

train_losses = []
val_accs = []
best_val_acc = 0.0
best_model_state = None

for epoch in range(EPOCHS):

    epoch_start = time.time()

    # Train
    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    
    # Evaluate
    val_acc = evaluate(val_loader, model)
    
    # Check if this is the best model so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save the state of the model in memory
        best_model_state = model.state_dict().copy()

    # Store metrics
    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Best Val: {best_val_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")

# --------------------------------------------------------------
# Post-Training & Plotting
# --------------------------------------------------------------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

# Load the best model weights to report Test Accuracy
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights based on Validation Accuracy.")

final_test_acc = evaluate(test_loader, model)
print(f"Test Accuracy (of model with best Val Acc): {final_test_acc:.4f}")

# Config string for filenames
config = f"lr{LEARNING_RATE}_bs{BATCH_SIZE}_epochs{EPOCHS}"

# Plotting
# Convert range to list for plotting
epoch_list = list(range(1, EPOCHS + 1))

plot(epoch_list, train_losses, ylabel='Training Loss', name=f'CNN-training-loss-{config}')
plot(epoch_list, val_accs, ylabel='Validation Accuracy', name=f'CNN-validation-accuracy-{config}')
# Test accuracy plot removed as per revised instructions

print("Plots saved to current directory.")
