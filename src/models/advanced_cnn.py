import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature importance"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, features = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        output = self.output(attended)
        
        return output, attention_weights

class AdvancedCNN(nn.Module):
    """Advanced 1D CNN with attention for IoT anomaly detection"""
    
    def __init__(self, input_features=115, num_classes=2, use_attention=True):
        super(AdvancedCNN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_features)
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3)
            )
        ])
        
        # Attention mechanism
        if self.use_attention:
            self.attention = AttentionLayer(256, 128)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Normalize input
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            x = x.transpose(1, 2)  # (batch, seq_len, features)
            x, attention_weights = self.attention(x)
            x = x.transpose(1, 2)  # Back to (batch, features, seq_len)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Classification
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'predictions': F.softmax(logits, dim=1),
            'attention_weights': attention_weights,
            'features': x
        }
    
    def get_model_size(self):
        return sum(p.numel() for p in self.parameters())

def create_advanced_cnn(config=None):
    if config is None:
        config = {'input_features': 115, 'num_classes': 2, 'use_attention': True}
    
    return AdvancedCNN(
        input_features=config.get('input_features', 115),
        num_classes=config.get('num_classes', 2),
        use_attention=config.get('use_attention', True)
    )

if __name__ == "__main__":
    print("🧠 Testing Advanced CNN with Attention...")
    
    model = create_advanced_cnn()
    print(f"Model created with {model.get_model_size():,} parameters")
    
    # Test with IoT data shape
    batch_size = 32
    test_input = torch.randn(batch_size, 115)
    
    output = model(test_input)
    
    print(f"✅ Input shape: {test_input.shape}")
    print(f"✅ Output logits: {output['logits'].shape}")
    print(f"✅ Output predictions: {output['predictions'].shape}")
    
    if output['attention_weights'] is not None:
        print(f"✅ Attention weights: {output['attention_weights'].shape}")
    
    print("🎉 Advanced model test successful!")
