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

# Save test dataset to text file for future use
test_list_file = "test_dataset_list.txt"
with open(test_list_file, "w") as f:
    for path, label_idx in test_samples:
        f.write(f"{path},{label_idx}\n")
print(f"Saved {len(test_samples)} test samples to {test_list_file}")


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

def load_test_dataset_from_txt(txt_file):
    test_samples = []
    with open(txt_file, "r") as f:
        for line in f:
            path_str, label_idx_str = line.strip().split(",")
            test_samples.append((Path(path_str), int(label_idx_str)))
    return test_samples

# Usage
test_samples = load_test_dataset_from_txt("test_dataset_list.txt")
test_dataset = KeywordSpottingDataset(test_samples)

