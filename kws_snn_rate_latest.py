# update the CUDA device
# dataset path update and download false
# update model, metrics and log filename

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torchaudio
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

from snntorch import spikegen, surrogate, functional as SF
import snntorch as snn

from snn_network import SNNRateConvNet
import utils
import custom_loss

# ----------------------------------------------------------------------------------------------------------------------------------------
class Logger(object):

    def __init__(self, log_file):
        self.terminal = sys.stdout  # Save reference to the console
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)  # Write to console
        self.log.write(message)       # Write to log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        

# === CONFIGURATION ===
batch_size = 128
num_epochs = 20
learning_rate = 7e-4
num_steps = 35
encoding_type = 'rate'  # or 'ttfs'
# encoding_type = 'ttfs'
load_model_path = f"saved_models/snn_conv2d_mel_{encoding_type}.pth"
save_model_path = f"saved_models/snn_conv2d_mel_{encoding_type}_reg.pth"
metrics_csv_path = f"logs/snn_conv2d_mel_{encoding_type}.csv"
training_log = f'logs/snn_conv2d_mel_{encoding_type}.log'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# # === DATASET ===
class KeywordSpottingDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label_idx = self.samples[index]
        waveform, sample_rate = torchaudio.load(path)
        label_name = path.parent.name  # Get the class name from the folder
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, sample_rate, label_name

def make_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    samples = []
    for target_class in class_to_idx:
        class_index = class_to_idx[target_class]
        target_dir = dataset_path / target_class
        for wav_path in target_dir.glob("*.wav"):
            samples.append((wav_path, class_index))
    return samples, class_to_idx

def split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_ratio, stratify=[s[1] for s in samples], random_state=42)
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_ratio/(train_ratio+val_ratio),
        stratify=[s[1] for s in train_val_samples], random_state=42)
    return train_samples, val_samples, test_samples

dataset_path = "../../../../datasets/kws_dataset"
samples, class_to_idx = make_dataset(dataset_path)
train_samples, val_samples, test_samples = split_dataset(samples)

# Create datasets
train_dataset = KeywordSpottingDataset(train_samples)
val_dataset = KeywordSpottingDataset(val_samples)
test_dataset = KeywordSpottingDataset(test_samples)

print("Class to index mapping:", class_to_idx)
print("Train/Val/Test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))

#-----------------------------------------------------------------------------------------------------------

# === Label Encoding ===
all_labels =  ["bed", "bird", "cat", "dog", "house", "marvin", "tree", "mask", "frame", "unknown", "silence"]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels) # encode labels as indices


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=64  # Recommended to avoid warnings
)
target_length = 16000

def collate_fn(batch):
    tensors, targets = [], []

    for waveform, sample_rate, label, *_ in batch:
        if waveform.size(1) < target_length:
            pad_size = target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:, :target_length]

        mel_spec = mel_transform(waveform).squeeze(0)  # Shape: [1, n_mels, time] squeezed to Shape: [n_mels, time] , useful for normalization
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        if encoding_type == 'ttfs':                
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6) # Min-max normalize to [0, 1] for latency encoding
        else:            
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5) # Mean-std normalization for rate encoding
        mel_spec = mel_spec.unsqueeze(0)  # Shape: [1, n_mels, time]
        tensors.append(mel_spec)  # [1, 64, 256]
        encoded_label = label_encoder.transform([label])[0]
        targets.append(encoded_label)

    return torch.stack(tensors), torch.tensor(targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === Label Info for Model Output ===
num_classes = len(class_to_idx)

# ----------------------------------------------------------------------------------------------------------------------------------------

#display
for data, label in train_loader:
  print(data[0].shape)                  # torch.Size([1, 64, 101])
  print(label[0])
  mel_spec = data[0].squeeze().numpy()  # shape: [64, 101] → [mel, time]
  break

# ----------------------------------------------------------------------------------------------------------------------------------------

# === LOSS & ACCURACY ===
if encoding_type == 'rate':
    loss_fn = SF.ce_rate_loss()
    accuracy_fn = SF.accuracy_rate
elif encoding_type == 'ttfs':
    loss_fn = SF.ce_temporal_loss()
    accuracy_fn = SF.accuracy_temporal
    

criterion = custom_loss.SpikeRegularizedLoss(loss_fn, initial_weight=0.0)   # weight is init to zero


# === TRAINING UTILS ===
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        spk_out = model(x, num_steps)
        loss = criterion(spk_out, y)
        # loss = loss_fn(spk_out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = accuracy_fn(spk_out, y)
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, return_sparsity=False):
    model.eval()
    total_loss = 0
    total_acc = 0
    accumulated_stats = {}

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if return_sparsity:
                spk_out, stats = model(x, num_steps, return_spike_stats=return_sparsity)
            else:
                spk_out= model(x, num_steps, return_spike_stats=return_sparsity)
            loss = loss_fn(spk_out, y)
            acc = accuracy_fn(spk_out, y)

            total_loss += loss.item()
            total_acc += acc
            
            if return_sparsity:
                """ sparsity - start """
                for layer, stats in stats.items():
                    if layer not in accumulated_stats:
                        accumulated_stats[layer] = {"spikes":0, "neurons": 0}
                    accumulated_stats[layer]["spikes"] += stats["spikes"]
                    accumulated_stats[layer]["neurons"] += stats["neurons"]*num_steps*y.size(0)
                """ sparsity - end """
    if return_sparsity:
        return total_loss / len(loader), total_acc / len(loader), accumulated_stats
    else:
        return total_loss / len(loader), total_acc / len(loader)

# ----------------------------------------------------------------------------------------------------------------------------------------

# === RUN TRAINING ===
model = SNNRateConvNet().to(device)
model.load_state_dict(torch.load(load_model_path, weights_only=True))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

metrics = []
best_val_loss = np.inf

# print("Start training")
sys.stdout = Logger(training_log)
print(model)
# test_loss, test_acc= evaluate(model, test_loader, return_sparsity=False)
test_loss, test_acc, old_sparsity= evaluate(model, test_loader, return_sparsity=True)
print(f"Before training, Test Loss: {test_loss:.4f}, Test Accuracy: {100*test_acc:.2f}%")

print(f'Start training: model with {encoding_type} encoding -- timesteps: {num_steps} -- learning rate: {learning_rate} -- epochs: {num_epochs}')

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {100*val_acc:.2f}%")
    metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    
    new_weight = custom_loss.update_weight_based_on_accuracy(val_acc, threshold_accuracy=0.8, target_weight=0.1)
    criterion.update_weight(new_weight)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"saved at epoch {epoch+1}")
        torch.save(model.state_dict(), save_model_path)

with open(metrics_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
# Save loss log
df = pd.DataFrame(metrics)
df.to_csv(metrics_csv_path, index=False)
# ----------------------------------------------------------------------------------------------------------------------------------------

# === FINAL TEST EVALUATION ===
model.load_state_dict(torch.load(save_model_path, weights_only=True))
test_loss, test_acc, new_sparsity = evaluate(model, test_loader, return_sparsity=True)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100*test_acc:.2f}%")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

old_sparsity = utils.compute_sparsity(old_sparsity)
new_sparsity = utils.compute_sparsity(new_sparsity)

utils.plot_line_graph_overlay(old_sparsity, new_sparsity, "logs/sparsity_rate_reg.png")

# utils.plot_training(encoding_type)