import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_features=115, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return x

def create_model():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    return model

if __name__ == "__main__":
    print("🧠 Testing CNN Model...")
    model = create_model()
    
    test_input = torch.randn(32, 115)
    output = model(test_input)
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print("🎉 Model works!")
