import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

import argparse
from logger import LoggerSetup
from src.data import convert_to_contiguous, preprocess_audio_batch
from src.modules import Attn_BN_Block

import types

def dict_to_class(name, d):
    cls = types.new_class(name, (object,))
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(cls, key, dict_to_class(key.capitalize(), value))
        else:
            setattr(cls, key, value)
    return cls

class TinySpeechZ1(nn.Module): 
    def __init__(self, num_classes, test=0): 
        super(TinySpeech, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block2 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block3 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=2, mid_channels_1=4, out_channels_1=7, test=test)
        self.block4 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=11, mid_channels_1=22, out_channels_1=7, test=test)
        self.block5 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=14, mid_channels_1=28, out_channels_1=7, test=test)
        self.block6 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=10, mid_channels_1=20, out_channels_1=7, test=test)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(204, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.test = test 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.cuda()

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

class TinySpeechZ2(nn.Module): 
    def __init__(self, num_classes, test=0): 
        super(TinySpeech, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block2 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block3 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=2, mid_channels_1=4, out_channels_1=7, test=test)
        self.block4 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=11, mid_channels_1=22, out_channels_1=7, test=test)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(204, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.test = test 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.cuda()

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

class TinySpeechZ3(nn.Module): 
    def __init__(self, num_classes, test=0): 
        super(TinySpeech, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block2 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(204, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.test = test 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.cuda()

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

if __name__ == "__main__": 
    logger = LoggerSetup("TinySpeech Trainer").get_logger()

    parser = argparse.ArgumentParser(description='Training script') 
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--quant', action='store_true', help='Quantize model')
    parser.add_argument('--prune', action='store_true', help='Prune model')
    parser.add_argument('--labels', type=int, default=31, help='Number of labels to train for')
    parser.add_argument('--save_pth', type=str, default=None, help='Path to save model after training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--quant_type', type=str, default="4bitsym", help='Precision level [2bitsym, 4bitsym, 8bit]')
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
            logger.info(f"Created directory {config.save_pth}")
    logger.info(f"Model weights to be saved at {config.save_pth}")

    # Source: https://huggingface.co/datasets/google/speech_commands
    labels = ["Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow", "__silence__"]
    num_labels = len(config.labels) # Choose how many target labels you want to train on 
    indices_labels = [labels.index(label) for label in config.labels]
    speech = load_dataset(config.dataset, config.version, config.split, trust_remote_code=True)
    speech = speech.shuffle()

    if num_labels != 31: 
        speech = speech.filter(lambda example: example['label'] in indices_labels)
        lut = convert_to_contiguous(indices_labels)
        speech = speech.map(lambda example: {"audio": example["audio"], "label": lut[example["label"]]})
        logger.info(f"Filtered dataset to {num_labels} labels")
    
    dataloader = torch.utils.data.DataLoader(speech, batch_size=config.batch_size, collate_fn=preprocess_audio_batch)

    if config.quant:
        from src.tinyspeech import QTinySpeechZ
        model = QTinySpeechZ(num_classes=num_labels)
    else:
        pass 
        # model = TinySpeechZ(num_classes=num_labels)

    # The proposed TinySpeech networks were trained using the SGD
    # optimizer in TensorFlow with following hyperparameters: momentum=0.9, learning rate=0.01,
    # number of epochs=50, batch size=64.

    for model_type in [TinySpeechZ1, TinySpeechZ2, TinySpeechZ3]:
        model = model_type(num_classes=num_labels)
        model.train()  
        criterion = nn.CrossEntropyLoss()  
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        logger.info(f"Loading SGD optimizer with learning rate {config.lr} and momentum {config.momentum}")
        num_epochs = config.epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            # logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            batch_ = 0
            for inputs, targets in dataloader:
                outputs = model(inputs)  
                
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)  
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                batch_ += 1
                # logger.info(f"    Batch [{batch_}] -> Batch Loss: {loss.item():.4f} | Predicted Labels: {torch.unique(predicted)}")
            
            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            logger.critical(f"  Epoch [{epoch+1}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")

        torch.save(model.state_dict(), os.path.join(config.save_pth, "tinyspeech.pth"))
        logger.critical(f"Model saved at {os.path.join(config.save_pth, 'tinyspeech.pth')}")