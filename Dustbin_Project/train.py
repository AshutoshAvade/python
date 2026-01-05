# train.py
import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

# Replace classifier
model.classifier[1] = nn.Linear(1280, len(dataset.classes))
model.to(device)

# 🔒 Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

EPOCHS = 20

for epoch in range(EPOCHS):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] completed")

torch.save({
    "model": model.state_dict(),
    "classes": dataset.classes
}, "model.pth")

print("✅ Model training complete & saved as model.pth")
