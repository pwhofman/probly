"""test for temp scaling"""

import probly
from src.probly.calibration.temperature_scaling import torch as temp

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from probly.calibration import temperature_scaling
from probly.evaluation.metrics import expected_calibration_error



device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Load Data

transforms = T.Compose([T.ToTensor()])
train = torchvision.datasets.CIFAR10(root="~/datasets", train=True, download=True, transform=transforms)
train, cal = torch.utils.data.random_split(train, [0.8, 0.2])
test = torchvision.datasets.CIFAR10(root="~/datasets", train=False, download=True, transform=transforms)
train_loader = DataLoader(train, batch_size=256, shuffle=True)
cal_loader = DataLoader(cal, batch_size=256, shuffle=True)
test_loader = DataLoader(test, batch_size=256, shuffle=False)

#Load Network

net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 10, device=device)
net = net.to(device)

# Train Model

epochs = 5
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in tqdm(range(epochs)):
    net.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Running loss: {running_loss / len(train_loader)}")




# compute accuracy and expected calibration error on test set
net.eval()
with torch.no_grad():
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for inpt, target in tqdm(test_loader):
        outputs = torch.cat((outputs, net(inpt.to(device))), dim=0)
        targets = torch.cat((targets, target.to(device)), dim=0)
logits = outputs                                    # save logits for calibration
outputs = F.softmax(outputs, dim=1)
correct = torch.sum(torch.argmax(outputs, dim=1) == targets).item()
total = targets.size(0)



probs = torch.softmax(logits, dim=1)

calibrated_model = temp.calibrate_temperature_grid(logits, targets)
calibrated_probs = temp.apply_temperature_scaling_probs(probs, calibrated_model)



ece1 = expected_calibration_error(outputs.cpu().numpy(), targets.cpu().numpy(), num_bins=10)
ece2 = expected_calibration_error(calibrated_probs.cpu().detach().numpy(), targets.cpu().numpy(), num_bins=10)

print(f"Accuracy: {correct / total}")
print(f"Expected Calibration Error: {ece1}")
print(f"Expected Calibration Error: {ece2}")








