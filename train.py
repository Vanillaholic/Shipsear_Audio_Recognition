import torchvision
import torch 
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchmetrics
import librosa  
import numpy as np
import argparse
import swanlab
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os



'''定义数据集'''
class ShipsearSet(Dataset):
    def __init__(self, data_root, num_classes):
        super().__init__()
        self.num_classes = num_classes
        relative_path = "E:\\MTQP\\wjy_codes\\shipsear_5s_16k\\"
        self.data = []
        with open(data_root, 'r') as f:
            for line in f:
                tmp = line.removeprefix(relative_path)[6:-1]
                wav, label = tmp.split('\t')
                self.data.append((os.path.join(wav[0], wav[0:3], wav), int(label)))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        y, sr = librosa.load(self.data[idx][0])
        return torch.tensor(y), self.data[idx][1]

'''定义梅尔谱变换器'''
mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,      # 采样率
    n_fft=1024,            # FFT窗口大小
    hop_length=512,         # 帧移
    n_mels=128,            # 梅尔滤波器组数量
    f_min=0,               # 最小频率
    f_max=8000,            # 最大频率
    window_fn=torch.hann_window  # 窗口函数
)

'''定义模型'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=10, padding=5, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=4)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        self.faltten = nn.Flatten()
        self.MLP = nn.Sequential(
            nn.Linear(432, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x)
        # 线性层
        x = self.faltten(x)
        x = self.MLP(x)
        #x = nn.functional.softmax(x, dim=1)
        return x

'''训练函数'''
def train(net, train_dataLoader, mel_spectrogram, optimizer, loss_criterion, device):
    """训练一个epoch"""
    net.train()
    total_loss = 0
    num_batches = 0
    
    for data in train_dataLoader:
        optimizer.zero_grad()
        wavs, labels = data
        wavs, labels = wavs.to(device), labels.to(device)

        mel_spec = mel_spectrogram(wavs)
        mel_spec = mel_spec.unsqueeze(1)

        preds = net(mel_spec)
        loss = loss_criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

'''测试函数'''
def test(net, test_dataLoader, mel_spectrogram, device):
    """测试模型"""
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_dataLoader:
            wav, labels = data
            wav, labels = wav.to(device), labels.to(device)
            mel_spec = mel_spectrogram(wav)
            mel_spec = mel_spec.unsqueeze(1)
            output = net(mel_spec)
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

if __name__ == "__main__":

    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备{device}")
    
    train_dataset = ShipsearSet(data_root="train_list.txt", num_classes=5)
    test_dataset = ShipsearSet(data_root="test_list.txt", num_classes=5)
    
    train_dataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    # 初始化模型

    net = Net().to(device)
    mel_spectrogram = mel_spectrogram.to(device)
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_criterion = nn.CrossEntropyLoss()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    
    # 更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    
    # 初始化SwanLab
    run = swanlab.init(
        project="Shipsear_5s_16k",
        experiment_name="第2次实验",
        config=args
    )
    
    # 训练循环

    for epoch in tqdm(range(args.num_epoch), desc="Training", unit="epoch"):
        # 训练
        loss = train(net, train_dataLoader, mel_spectrogram, optimizer, loss_criterion, device)
        
        # 测试
        acc = test(net, test_dataLoader, mel_spectrogram, device)
        
        # 记录到SwanLab
        run.log({"Loss": loss,"Accuracy": acc})
        
        # 打印进度
        tqdm.write(f"Epoch {epoch+1}/{args.num_epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
    

    
    # 保存模型
    torch.save(net.state_dict(), 'final_model.pth')
    print("模型已保存为 'final_model.pth'")

