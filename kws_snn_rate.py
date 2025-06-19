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
from sklearn.metrics import confusion_matrix
import seaborn as sn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

from snntorch import spikegen, surrogate, functional as SF
import snntorch as snn

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
num_epochs = 50
learning_rate = 1e-3
num_steps = 35
encoding_type = 'rate'  # or 'ttfs'
# encoding_type = 'ttfs'
save_model_path = f"saved_models/snn_conv2d_mel_{encoding_type}.pth"
metrics_csv_path = f"logs/snn_conv2d_mel_{encoding_type}.csv"
training_log = f'logs/snn_conv2d_mel_{encoding_type}.log'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------------------------------

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

dataset_path = "../../../datasets/kws_dataset"
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
    n_mels=40  # Recommended to avoid warnings
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
        #   mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        if encoding_type == 'ttfs':
                # Min-max normalize to [0, 1] for latency encoding
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        else:
            # Mean-std normalization for rate encoding
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
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

def print_confusion_matrix(y_true, y_pred):
  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in all_labels],
                      columns = [i for i in all_labels])
  plt.figure(figsize = (12,7))
  sn.heatmap(df_cm, annot=True)
  fig_path = 'logs/kws_snn_cm.png'
  plt.savefig(fig_path)

def test(model):
    correct=0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    #confusion matrix
    y_pred = []
    y_true = []

    model.eval() # test the model with dropout layers off
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        output = model(images, num_steps)

        _, pred = output.sum(dim=0).max(1)
        # correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct += correct_tensor.sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct_tensor[i].item()
            class_total[label] += 1

    # Print overall accuracy
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')
    # Print per-class accuracy
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'Accuracy of class {all_labels[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of class {all_labels[i]}: N/A (no samples)')
    print_confusion_matrix(y_true, y_pred)

# ----------------------------------------------------------------------------------------------------------------------------------------


# === MODEL ===
class SNNConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        beta = 0.95
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(128 * 10 * 13, num_classes)
        self.fc1 = nn.Linear(64*6, 512) 
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(512, 128) 
        self.lif5 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(128, num_classes) 
        self.lif6 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        spk_out_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        cur_input = encode_input(x, num_steps)
        for step in range(num_steps):
            cur1 = self.conv1(cur_input[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.conv2(spk1)
            cur2 = self.pool(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.conv3(spk2)
            cur3 = self.pool(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            flat = self.flatten(spk3)
            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)
            
            spk_out_rec.append(spk6)

        return torch.stack(spk_out_rec)
    
# ----------------------------------------------------------------------------------------------------------------------------------------

# === ENCODING ===
def encode_input(x, num_steps):
    if encoding_type == 'rate':
        return spikegen.rate(x, num_steps=num_steps)
    elif encoding_type == 'ttfs':
        return spikegen.latency(x, num_steps=num_steps, normalize=True, linear=True)
    else:
        raise ValueError("Unknown encoding type")

# === LOSS & ACCURACY ===
if encoding_type == 'rate':
    loss_fn = SF.ce_rate_loss()
    accuracy_fn = SF.accuracy_rate
elif encoding_type == 'ttfs':
    loss_fn = SF.ce_temporal_loss()
    accuracy_fn = SF.accuracy_temporal

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
        loss = loss_fn(spk_out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = accuracy_fn(spk_out, y)
        total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            spk_out = model(x, num_steps)
            loss = loss_fn(spk_out, y)
            acc = accuracy_fn(spk_out, y)

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(loader), total_acc / len(loader)

# ----------------------------------------------------------------------------------------------------------------------------------------

# === RUN TRAINING ===
model = SNNConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

metrics = []
best_val_loss = np.inf

# print("Start training")
sys.stdout = Logger(training_log)
print(model)
print(f'Start training: model with {encoding_type} encoding -- timesteps: {num_steps} -- learning rate: {learning_rate} -- epochs: {num_epochs}')

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {100*val_acc:.2f}%")
    metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_model_path)

with open(metrics_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
# Save loss log
df = pd.DataFrame(metrics)
df.to_csv(metrics_csv_path, index=False)

# ----------------------------------------------------------------------------------------------------------------------------------------

# === FINAL TEST EVALUATION ===
model.load_state_dict(torch.load(save_model_path))
test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100*test_acc:.2f}%")
test(model)

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

# ----------------------------------------------------------------------------------------------------------------------------------------

# log = pd.read_csv(metrics_csv_path)
log = pd.read_csv(f"logs/snn_conv2d_mel_{encoding_type}.csv")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
plt.plot(log["epoch"], log["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(log["epoch"], log["train_acc"], label="Train Accuracy")
plt.plot(log["epoch"], log["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig(f"logs/snn_conv2d_mel_{encoding_type}_plot.png")
print(f"Plot saved as logs/snn_conv2d_mel_{encoding_type}_plot.png")
plt.show()
