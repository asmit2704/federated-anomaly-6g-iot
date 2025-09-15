import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

class IoTDataset(data.Dataset):
    """Generic IoT dataset for anomaly detection"""
    
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

class DatasetLoader:
    """Main class for loading and preprocessing IoT datasets"""
    
    def __init__(self, dataset_name: str = 'n_baiot'):
        self.dataset_name = dataset_name
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_synthetic_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic IoT data for testing"""
        print(f"🔄 Generating synthetic {self.dataset_name} data...")
        
        # Generate features based on dataset type
        if self.dataset_name == 'n_baiot':
            features = np.random.randn(num_samples, 115)
            # Add some patterns for anomalies
            anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.1), replace=False)
            features[anomaly_indices] *= 3  # Make anomalies more extreme
            
        elif self.dataset_name == 'bot_iot':
            features = np.random.randn(num_samples, 43)
            anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.15), replace=False)
            features[anomaly_indices] += np.random.randn(len(anomaly_indices), 43) * 2
            
        elif self.dataset_name == 'ton_iot':
            features = np.random.randn(num_samples, 47)
            anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.12), replace=False)
            features[anomaly_indices] *= 2.5
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Generate labels (0 = normal, 1 = anomaly)
        labels = np.zeros(num_samples)
        labels[anomaly_indices] = 1
        
        print(f"✅ Generated {num_samples} samples with {len(anomaly_indices)} anomalies")
        return features, labels
    
    def preprocess_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and labels"""
        print("🔄 Preprocessing data...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"✅ Features scaled to range [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
        print(f"✅ Labels encoded: {np.unique(labels_encoded)}")
        
        return features_scaled, labels_encoded
    
    def create_dataloaders(self, batch_size: int = 64, test_split: float = 0.2, 
                          num_samples: int = 10000) -> Tuple[data.DataLoader, data.DataLoader]:
        """Create train and test dataloaders"""
        
        # Load data (synthetic for now)
        features, labels = self.load_synthetic_data(num_samples)
        
        # Preprocess
        features, labels = self.preprocess_data(features, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_split, random_state=42, stratify=labels
        )
        
        print(f"📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        print(f"📊 Normal/Anomaly ratio in train: {(y_train==0).sum()}:{(y_train==1).sum()}")
        
        # Create datasets
        train_dataset = IoTDataset(X_train, y_train)
        test_dataset = IoTDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return train_loader, test_loader

def get_dataset_loader(dataset_name: str = 'n_baiot', batch_size: int = 64, 
                      test_split: float = 0.2, num_samples: int = 10000):
    """Factory function to get dataset loader"""
    loader = DatasetLoader(dataset_name)
    return loader.create_dataloaders(batch_size, test_split, num_samples)

if __name__ == "__main__":
    print("📊 Testing Dataset Loader...")
    
    # Test with different datasets
    for dataset in ['n_baiot', 'bot_iot', 'ton_iot']:
        print(f"\n🔍 Testing {dataset} dataset:")
        
        train_loader, test_loader = get_dataset_loader(
            dataset_name=dataset, 
            batch_size=32, 
            num_samples=1000
        )
        
        # Test one batch
        for features, labels in train_loader:
            print(f"✅ Batch shape: {features.shape}, Labels: {labels.shape}")
            print(f"✅ Feature range: [{features.min():.2f}, {features.max():.2f}]")
            print(f"✅ Unique labels: {labels.unique()}")
            break
    
    print("🎉 Dataset loader test successful!")
