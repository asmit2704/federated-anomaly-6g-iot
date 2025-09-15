import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.advanced_cnn import create_advanced_cnn

class EdgeOptimizedStudent(nn.Module):
    """Ultra-lightweight student model for IoT edge devices"""
    
    def __init__(self, input_features=115, num_classes=2):
        super(EdgeOptimizedStudent, self).__init__()
        
        # Minimal architecture for edge deployment
        self.input_norm = nn.BatchNorm1d(input_features)
        
        # Single lightweight conv layer
        self.conv = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(8)  # Fixed small size
        
        # Tiny classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_norm(x).unsqueeze(1)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'predictions': F.softmax(logits, dim=1),
            'features': x
        }
    
    def get_model_size(self):
        return sum(p.numel() for p in self.parameters())

def run_distillation(dataset_name='n_baiot'):
    """Run complete knowledge distillation"""
    print("📚 Knowledge Distillation for IoT Edge Deployment")
    
    # Create teacher (your advanced model)
    teacher = create_advanced_cnn()
    print(f"👨‍🏫 Teacher: {teacher.get_model_size():,} parameters")
    
    # Create student (ultra-light)
    student = EdgeOptimizedStudent()
    print(f"👨‍🎓 Student: {student.get_model_size():,} parameters")
    
    compression = teacher.get_model_size() / student.get_model_size()
    print(f"🗜️  Compression: {compression:.1f}x smaller")
    
    # Load data
    from utils.data_loader import get_dataset_loader
    train_loader, test_loader = get_dataset_loader(dataset_name, batch_size=32, num_samples=1000)
    
    # Distillation training (simplified)
    print("🔄 Training student model...")
    
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    teacher.eval()
    student.train()
    
    for epoch in range(3):
        for batch_idx, (data, labels) in enumerate(train_loader):
            if batch_idx >= 5:  # Quick training
                break
                
            optimizer.zero_grad()
            
            # Teacher predictions (frozen)
            with torch.no_grad():
                teacher_output = teacher(data)
                teacher_soft = F.softmax(teacher_output['logits'] / 4.0, dim=1)  # Temperature = 4
            
            # Student predictions
            student_output = student(data)
            student_soft = F.log_softmax(student_output['logits'] / 4.0, dim=1)
            
            # Distillation loss
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            hard_loss = F.cross_entropy(student_output['logits'], labels)
            
            total_loss = 0.7 * kl_loss * 16 + 0.3 * hard_loss  # Alpha=0.7, temp^2=16
            total_loss.backward()
            optimizer.step()
    
    print("✅ Distillation training complete!")
    
    # Evaluate both models
    teacher.eval()
    student.eval()
    
    teacher_acc = 0
    student_acc = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            teacher_pred = teacher(data)['logits'].argmax(dim=1)
            student_pred = student(data)['logits'].argmax(dim=1)
            
            teacher_acc += (teacher_pred == labels).sum().item()
            student_acc += (student_pred == labels).sum().item()
            total += labels.size(0)
            
            if total >= 200:  # Limit evaluation
                break
    
    teacher_accuracy = teacher_acc / total
    student_accuracy = student_acc / total
    
    results = {
        'teacher_accuracy': teacher_accuracy,
        'student_accuracy': student_accuracy,
        'accuracy_retention': student_accuracy / teacher_accuracy,
        'compression_ratio': compression,
        'teacher_params': teacher.get_model_size(),
        'student_params': student.get_model_size()
    }
    
    print(f"\n📊 Distillation Results:")
    print(f"   👨‍🏫 Teacher Accuracy: {teacher_accuracy:.4f}")
    print(f"   👨‍🎓 Student Accuracy: {student_accuracy:.4f}")
    print(f"   🎯 Accuracy Retention: {results['accuracy_retention']:.1%}")
    print(f"   🗜️  Model Compression: {compression:.1f}x")
    
    return student, results

if __name__ == "__main__":
    student_model, results = run_distillation()
    print("✅ Knowledge distillation test complete!")
