import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk, concatenate_datasets
import matplotlib.pyplot as plt
import os
import argparse
from logger import LoggerSetup
from training.data import convert_to_contiguous, preprocess_audio_batch
from training.modules import QModel 
from training.tinyspeech import TinySpeechZ, QTinySpeechZ
from training.exportquant import export_to_hfile
import types
from tqdm import tqdm

import faulthandler

faulthandler.enable()

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def plot_metrics(loss_values, accuracy_values, model_name, save_path):
    epochs = range(1, len(loss_values) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'r', label='Loss')
    plt.title(f'Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_values, 'b', label='Accuracy')
    plt.title(f'Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

def dict_to_class(name, d):
    cls = types.new_class(name, (object,))
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(cls, key, dict_to_class(key.capitalize(), value))
        else:
            setattr(cls, key, value)
    return cls

class TinySpeechZ1(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "TinySpeechZ-1"
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x

class TinySpeechZ2(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "TinySpeechZ-2"
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x

class TinySpeechZ3(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

    def __str__(self): 
        return "TinySpeechZ-3"

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x



if __name__ == "__main__": 
    logger = LoggerSetup("TinySpeech Trainer").get_logger()

    parser = argparse.ArgumentParser(description='Training script') 
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--quant', action='store_true', help='Quantize model')
    parser.add_argument('--model_type',type=str, default='Z', help='Model Type')
    parser.add_argument('--prune', action='store_true', help='Prune model')
    parser.add_argument('--labels', type=int, default=31, help='Number of labels to train for')
    parser.add_argument('--save_pth', type=str, default=None, help='Path to save model after training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--quant_type', type=int, default=4, help='Precision level [2bitsym, 4bitsym, 8bit]')
    parser.add_argument('--wscale', type=str, default="PerTensor", help='Seed for reproducibility')
    parser.add_argument('--quant_scale', type=float, default=0.25, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on') 
    args = parser.parse_args()

    if args.config: 
        logger.info(f"Loading config from {args.config}")
        import yaml
        with open(args.config, 'r') as f:
            config = dict_to_class('config', yaml.safe_load(f))
    else: 
        config = args

    torch.manual_seed(config.seed)
    logger.info(f"Setting seed to {config.seed}")

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
    logger.info(f"Training on {config.device}")

    import os
    if config.save_pth:
        if not os.path.exists(config.save_pth):
            os.makedirs(config.save_pth)
            logger.info(f"Created directory '{config.save_pth}'")
    logger.info(f"Model weights to be saved at '{config.save_pth}'")

    # Source: https://huggingface.co/datasets/google/speech_commands
    labels = ["Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow", "__silence__"]
    num_labels = len(config.labels) # Choose how many target labels you want to train on 
    indices_labels = [labels.index(label) for label in config.labels]
    if not hasattr(config.dataset, 'path'):
        speech_train = load_dataset(config.dataset.name, config.dataset.version, split='train', trust_remote_code=True)
        speech_validation = load_dataset(config.dataset.name, config.dataset.version, split='validation', trust_remote_code=True)
        speech_test = load_dataset(config.dataset.name, config.dataset.version, split='test', trust_remote_code=True)
        logger.info(f"Loaded dataset from {config.dataset.name} v{config.dataset.version}")
        speech = concatenate_datasets([speech_train, speech_validation, speech_test])
        speech = speech.shuffle()

        if num_labels != 31: 
            speech = speech.filter(lambda example: example['label'] in indices_labels)
            lut = convert_to_contiguous(indices_labels)
            speech = speech.map(lambda example: {"audio": example["audio"], "label": lut[example["label"]]})
            logger.info(f"Filtered dataset to {num_labels} labels")
            speech.save_to_disk('./data/filtered')  
    else: 
        speech = load_from_disk(config.dataset.path)
        logger.info(f"Loaded dataset from {config.dataset.path}")
    
    dataloader = torch.utils.data.DataLoader(speech, batch_size=config.batch_size, collate_fn=preprocess_audio_batch)

    if config.quant:
        logger.info(f"Loading Quantized Version!")
        if config.model_type == "Z": 
            if config.quant_type < 8: 
                model = QTinySpeechZ(num_classes=num_labels, QuantType=f"{config.quant_type}bitsym")
            else: 
                model = QTinySpeechZ(num_classes=num_labels, QuantType=f"{config.quant_type}bit")
    else:
        logger.info(f"Loading Unquantized Version!")
        if config.model_type == "Z": 
            model = TinySpeechZ(num_classes=num_labels).to(torch.device("cuda")) 

    # The proposed TinySpeech networks were trained using the SGD
    # optimizer in TensorFlow with following hyperparameters: momentum=0.9, learning rate=0.01,
    # number of epochs=50, batch size=64.

    logger.info(f"Model loaded with {numel(model)} parameters") 
    logger.info(f"Model size is {config.quant_type * numel(model, only_trainable=True) / 1000} kb")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=1e-4)
    logger.info(f"Loading SGD optimizer with learning rate {config.lr} and momentum {config.momentum}")
    
    num_epochs = config.epochs
    loss_values = []
    accuracy_values = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for inputs, targets in progress_bar:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), accuracy=correct_predictions/total_samples)

        progress_bar.close()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_accuracy)


    plot_metrics(loss_values, accuracy_values, str(model), os.path.join(config.save_pth, str(model)))
    logger.critical(f"Model plots saved at {os.path.join(config.save_pth, str(model))}")

    torch.save(model.state_dict(), os.path.join(config.save_pth, str(model) + '.pth'))
    logger.critical(f"Model saved at {os.path.join(config.save_pth, str(model))}")

    if config.quant: 
        model = QModel(model)
        logger.info(f"Quantized model loaded with {model.numel()[0]} parameters")

        model.quantize() 
        model.save_checkpoint(os.path.join(config.save_pth, str(model) + '_quant.pth'))

        export_to_hfile(model, os.path.join(config.save_pth, str(model) + '.h'), str(model))

