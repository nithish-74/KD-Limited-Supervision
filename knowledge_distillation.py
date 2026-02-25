import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 1. Define Teacher and Student Models ---
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Training Functions ---
def train_teacher(model, train_loader, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Teacher Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

def train_student_kd(teacher, student, train_loader, optimizer, device, T, alpha, epochs=5):
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            student_logits = student(data)
            
            # Soft targets loss
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            ) * (T * T)
            
            # Hard targets loss
            hard_loss = F.cross_entropy(student_logits, target)
            
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Student KD Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

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

# --- 3. Main Experiment ---
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 1. Train Teacher on FULL dataset
    teacher = TeacherNet().to(device)
    optimizer_t = optim.Adam(teacher.parameters(), lr=0.001)
    print("\nTraining Teacher Model (Full Dataset)...")
    train_teacher(teacher, train_loader, optimizer_t, device, epochs=10)
    teacher_acc = test(teacher, test_loader, device)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")

    # --- Low-Data Experiment Setup ---
    print("\n--- Low-Data Experiment (20% Labels) ---")
    num_samples = len(train_dataset)
    indices = np.random.choice(num_samples, int(0.2 * num_samples), replace=False)
    reduced_train_dataset = Subset(train_dataset, indices)
    low_data_loader = DataLoader(reduced_train_dataset, batch_size=64, shuffle=True)
    print(f"Reduced training set size: {len(reduced_train_dataset)} samples")

    # 2. Train Student from scratch on REDUCED dataset
    student_scratch = StudentNet().to(device)
    optimizer_s1 = optim.Adam(student_scratch.parameters(), lr=0.001)
    print("\nTraining Student from Scratch (20% data)...")
    # Using simple CE loss
    student_scratch.train()
    for epoch in range(10):
        running_loss = 0.0
        for data, target in low_data_loader:
            data, target = data.to(device), target.to(device)
            optimizer_s1.zero_grad()
            loss = F.cross_entropy(student_scratch(data), target)
            loss.backward()
            optimizer_s1.step()
            running_loss += loss.item()
        print(f"Student (Scratch) Epoch {epoch+1}, Loss: {running_loss/len(low_data_loader):.4f}")
    scratch_acc = test(student_scratch, test_loader, device)
    print(f"Student (Scratch) Accuracy (20% data): {scratch_acc:.2f}%")

    # 3. Train Student with Knowledge Distillation on REDUCED dataset
    student_kd = StudentNet().to(device)
    optimizer_s2 = optim.Adam(student_kd.parameters(), lr=0.001)
    print("\nTraining Student with Knowledge Distillation (T=10, Alpha=0.9, 20% data)...")
    train_student_kd(teacher, student_kd, low_data_loader, optimizer_s2, device, T=10, alpha=0.9, epochs=10)
    kd_acc = test(student_kd, test_loader, device)
    print(f"Student (KD) Accuracy (20% data): {kd_acc:.2f}%")

    # --- Results Summary ---
    print("\n--- Summary (Low-Data) ---")
    print(f"Teacher Accuracy (Full): {teacher_acc:.2f}%")
    print(f"Student (Scratch) Accuracy (20%): {scratch_acc:.2f}%")
    print(f"Student (KD) Accuracy (20%): {kd_acc:.2f}%")
    print(f"Improvement with KD in Low-Data setting: {kd_acc - scratch_acc:.2f}%")

    # Plotting results
    labels = ['Teacher (Full)', 'Student Scratch (20%)', 'Student KD (20%)']
    accuracies = [teacher_acc, scratch_acc, kd_acc]
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, accuracies, color=['blue', 'red', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title('Low-Data Knowledge Distillation: 20% Training Data Labels (T=10, Alpha=0.9)')
    plt.ylim(85, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
    plt.savefig('kd_low_data_results.png')
    print("\nPlot saved as 'kd_low_data_results.png'")

if __name__ == "__main__":
    run_experiment()
