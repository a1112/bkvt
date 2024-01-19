import torch
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.data import create_transform

from BKVisionTrain.git.NoisyVIT.model import NoisyViT

# 数据加载和预处理
# 以下代码需要根据您的具体数据集进行调整
transform = create_transform(input_size=224, is_training=True)
dataset = datasets.ImageFolder(root=r'E:\clfData\data', transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 模型初始化
model = NoisyViT(optimal=True, res=224, num_classes=1000)  # 假设有1000个类别
model = model.cuda()

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200

print(model)

# 训练循环
model.train()
for epoch in range(num_epochs):
    print(epoch)
    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 保存模型
torch.save(model.state_dict(), 'noisy_vit_model.pth')
