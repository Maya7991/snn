device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

all_labels =  ["bed", "bird", "cat", "dog", "house", "marvin", "tree", "mask", "frame", "unknown", "silence"]
dataset_path = "../../datasets/kws_dataset"
batch_size = 128
encoding_type = "rate"
num_steps = 35

# ----------------------------------------------------------------------------------------------------------------------------------------

# Load the test data
train_loader, val_loader, test_loader, label_encoder, class_to_idx = get_data_loaders(
    dataset_path, all_labels, batch_size, encoding_type
)

for inputs, labels in test_loader:
    # input_shape = inputs[0].shape
    input_shape = tuple(inputs[0].shape)
    break

encoding_type = 'rate'

if encoding_type == 'rate':
    loss_fn = SF.ce_rate_loss()
    accuracy_fn = SF.accuracy_rate
elif encoding_type == 'ttfs':
    loss_fn = SF.ce_temporal_loss()
    accuracy_fn = SF.accuracy_temporal
    
# === ENCODING ===
def encode_input(x, num_steps):
    if encoding_type == 'rate':
        return spikegen.rate(x, num_steps=num_steps)
    elif encoding_type == 'ttfs':
        return spikegen.latency(x, num_steps=num_steps, normalize=True, linear=True)
    else:
        raise ValueError("Unknown encoding type")
    
    
# === MODEL ===
class SNNConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 11
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

    def forward(self, x, num_steps, return_spike_stats=False):
        spk_out_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        
        spike_stats = {
            "lif1": {"spikes": 0, "neurons": 0},
            "lif2": {"spikes": 0, "neurons": 0},
            "lif3": {"spikes": 0, "neurons": 0},
            "lif4": {"spikes": 0, "neurons": 0},
            "lif5": {"spikes": 0, "neurons": 0},
            "lif6": {"spikes": 0, "neurons": 0}
        }

        cur_input = encode_input(x, num_steps)
        for step in range(num_steps):
            cur1 = self.conv1(cur_input[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spike_stats["lif1"]["spikes"] += spk1.detach().sum().item()
            spike_stats["lif1"]["neurons"] = spk1[0].numel()
            
            cur2 = self.conv2(spk1)
            cur2 = self.pool(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_stats["lif2"]["spikes"] += spk2.detach().sum().item()
            spike_stats["lif2"]["neurons"] = spk2[0].numel()
            
            cur3 = self.conv3(spk2)
            cur3 = self.pool(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spike_stats["lif3"]["spikes"] += spk3.detach().sum().item()
            spike_stats["lif3"]["neurons"] = spk3[0].numel()
            
            flat = self.flatten(spk3)
            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, mem4)
            spike_stats["lif4"]["spikes"] += spk4.detach().sum().item()
            spike_stats["lif4"]["neurons"] = spk4[0].numel()
            
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            spike_stats["lif5"]["spikes"] += spk5.detach().sum().item()
            spike_stats["lif5"]["neurons"] = spk5[0].numel()
            
            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)
            spike_stats["lif6"]["spikes"] += spk6.detach().sum().item() # all batches
            spike_stats["lif6"]["neurons"] = spk6[0].numel()            # just 0th index sample
            
            spk_out_rec.append(spk6)
        if return_spike_stats:
            return torch.stack(spk_out_rec), spike_stats
        else:
            return torch.stack(spk_out_rec)

# Load the trained SNN model
snn_model = SNNConvNet()
snn_model.load_state_dict(torch.load('snn/saved_models/snn_conv2d_mel_rate.pth', weights_only=True))
snn_model.eval()

def test_snn(model, loader, device, all_labels, num_steps, encoding_type):
    model.to(device)
    correct=0
    total = 0
    num_classes = len(all_labels)
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    #confusion matrix
    y_pred = []
    y_true = []

    model.eval() # test the model with dropout layers off
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)        
        with torch.no_grad():
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

    accuracy = 100 * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')

    # Print per-class accuracy
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'Accuracy of class {all_labels[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of class {all_labels[i]}: N/A (no samples)')

    cm_save_path = f"logs/kws_snn_{encoding_type}_cm.png"
    utils.print_confusion_matrix(y_true, y_pred, all_labels, cm_save_path)
    
    
    return {
        "overall_accuracy": accuracy,
        "class_accuracy": {
            all_labels[i]: (100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else None
            for i in range(num_classes)
        },
        "confusion_matrix_path": cm_save_path
    }
    
    
def test_snn_and_sparsity(model, loader, device, all_labels, num_steps, encoding_type):
    model.to(device)
    
    """ sparsity - start """
    accumulated_stats = {"total_samples":0}
    batch_count = 0
    """ sparsity - end """
    
    correct=0
    total = 0
    num_classes = len(all_labels)
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    y_pred = []
    y_true = []

    model.eval() # test the model with dropout layers off
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)        
        with torch.no_grad():
            output, stats = model(images, num_steps, return_spike_stats=True)
        _, pred = output.sum(dim=0).max(1)
        
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct += correct_tensor.sum().item()
        total += labels.size(0)
        
        """ sparsity - start """
        for layer, stats in stats.items():
            if layer not in accumulated_stats:
                accumulated_stats[layer] = {"spikes":0, "neurons": 0}
            accumulated_stats[layer]["spikes"] += stats["spikes"]
            accumulated_stats[layer]["neurons"] += stats["neurons"]*num_steps*images.size(0)            
        accumulated_stats["total_samples"] += images.size(0)
        """ sparsity - end """

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct_tensor[i].item()
            class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')

    cm_save_path = f"logs/kws_snn_{encoding_type}_cm.png"
    utils.print_confusion_matrix(y_true, y_pred, all_labels, cm_save_path)
        
    return {
        "overall_accuracy": accuracy,
        "class_accuracy": {
            all_labels[i]: (100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else None
            for i in range(num_classes)
        },
        "confusion_matrix_path": cm_save_path,
        "accumulated_stats": accumulated_stats
    }
    
def compute_sparsity(accumulated_stats):
    layers = list(accumulated_stats.keys())
    sparsity_layers = {}
    for layer in layers:
        spikes = accumulated_stats[layer]["spikes"]
        neurons = accumulated_stats[layer]["neurons"]
        total_possible_spikes = neurons
        sparsity = 1-(spikes / total_possible_spikes)
        sparsity_layers[layer] = (sparsity*100)
    return sparsity_layers