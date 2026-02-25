import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Data Preparation ---
def get_data_loaders(num_labeled=1000, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split into labeled and unlabeled
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]

    labeled_loader = DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(Subset(train_dataset, unlabeled_indices), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return labeled_loader, unlabeled_loader, test_loader

# --- 3. Pseudo-Labeling Training ---
def train_pseudo_labeling(model, labeled_loader, unlabeled_loader, test_loader, device, epochs=5, threshold=0.9):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # 1. Train on labeled data
        for data, target in labeled_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 2. Pseudo-labeling step (on a batch of unlabeled data)
        model.eval()
        unlabeled_batch = next(iter(unlabeled_loader))
        unlabeled_data, _ = unlabeled_batch
        unlabeled_data = unlabeled_data.to(device)
        
        with torch.no_grad():
            output = model(unlabeled_data)
            probs = F.softmax(output, dim=1)
            max_probs, pseudo_labels = torch.max(probs, 1)
            
            # Filter by threshold
            mask = max_probs > threshold
        
        if mask.sum() > 0:
            filtered_data = unlabeled_data[mask]
            filtered_labels = pseudo_labels[mask]
            
            # Re-train with pseudo-labels
            model.train()
            optimizer.zero_grad()
            output_pseudo = model(filtered_data)
            loss_pseudo = F.cross_entropy(output_pseudo, filtered_labels)
            loss_pseudo.backward()
            optimizer.step()
        
        acc = test(model, test_loader, device)
        accuracies.append(acc)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(labeled_loader):.4f}, Test Acc: {acc:.2f}%")
        
    return accuracies

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

# --- 4. Comparison with Supervised Only ---
def train_supervised_only(model, labeled_loader, test_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data, target in labeled_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        acc = test(model, test_loader, device)
        accuracies.append(acc)
        print(f"Supervised Epoch {epoch+1}, Test Acc: {acc:.2f}%")
        
    return accuracies

def run_semi_supervised_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(num_labeled=500) # Only 500 labeled examples

    print("\n--- Training Supervised Baseline (500 labels) ---")
    model_sup = SimpleCNN().to(device)
    sup_accs = train_supervised_only(model_sup, labeled_loader, test_loader, device, epochs=5)

    print("\n--- Training Semi-Supervised (Pseudo-Labeling) ---")
    model_semi = SimpleCNN().to(device)
    semi_accs = train_pseudo_labeling(model_semi, labeled_loader, unlabeled_loader, test_loader, device, epochs=5)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), sup_accs, label='Supervised Only (500 labels)', marker='o')
    plt.plot(range(1, 6), semi_accs, label='Pseudo-Labeling (Semi-Supervised)', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Semi-Supervised Learning Impact with Limited Labeled Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('semi_supervised_results.png')
    print("\nPlot saved as 'semi_supervised_results.png'")

if __name__ == "__main__":
    run_semi_supervised_experiment()
