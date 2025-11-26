"""Implementation of temperature scaling in torch."""

import torch
import torch.nn as nn
import torch.nn.functional as F



# Suppose val_probs, val_labels


#val_logits = torch.log(val_probs + 1e-12)  # pseudo logits
#val_labels = val_labels.long()




class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        # adapted to run on same device for later comparisons
        temperature = self.temperature.to(logits.device)
        return logits / temperature
        #return logits / self.temperature

def calibrate_temperature_grid(logits, labels, temp_min=0.5, temp_max=5.0, num_steps=100):
    """
    Find optimal temperature T using grid search
    """
    scaler = TempScaler()  # use original module
    best_temp = 1.0
    best_loss = float('inf')
    labels = labels.long()
    
    temperatures = torch.linspace(temp_min, temp_max, num_steps)
    
    for T in temperatures:
        # improve the scaler's temperature
        scaler.temperature.data.fill_(T)
        scaled_logits = scaler.forward(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        if loss < best_loss:
            best_loss = loss
            best_temp = T.item()
    
    # set final optimal temperature
    scaler.temperature.data.fill_(best_temp)
    return scaler


def calibrate_temperature(logits, labels, max_iter=50, lr=0.01):
    scaler = TempScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = nn.functional.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def apply_temperature_scaling_probs(probs, scaler):
    pseudo_logits = torch.log(probs + 1e-12)       # convert probabilities to pseudo logits
    pseudo_logits = pseudo_logits.to(scaler.temperature.device)  # ensure same device
    scaled_logits = scaler(pseudo_logits)          # divide by learned temperature
    calibrated_probs = torch.softmax(scaled_logits, dim=1)
    return calibrated_probs

