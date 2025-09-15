import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.advanced_cnn import create_advanced_cnn
from utils.data_loader import get_dataset_loader

class IoTClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        print(f"🔄 Initializing IoT Client {client_id}...")
        
        self.model = create_advanced_cnn()
        print(f"📊 Model loaded: {self.model.get_model_size():,} parameters")
        
        # Load client data  
        self.train_loader, self.test_loader = get_dataset_loader(
            dataset_name='n_baiot', batch_size=32, num_samples=500
        )
        print(f"📁 Data loaded: {len(self.train_loader.dataset)} training samples")
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        print(f"🔄 Client {self.client_id}: Starting local training...")
        self.set_parameters(parameters)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Train for a few batches
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx >= 3:  # Quick training
                break
                
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output['logits'], target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"✅ Client {self.client_id}: Training complete, avg loss: {avg_loss:.4f}")
        
        return self.get_parameters(config), len(self.train_loader.dataset), {"train_loss": avg_loss}
    
    def evaluate(self, parameters, config):
        print(f"🔍 Client {self.client_id}: Evaluating model...")
        self.set_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if batch_idx >= 2:  # Quick evaluation
                    break
                    
                output = self.model(data)
                test_loss += criterion(output['logits'], target).item()
                pred = output['logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / min(2, len(self.test_loader))
        
        print(f"✅ Client {self.client_id}: Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return avg_loss, total, {"accuracy": accuracy}

# Create client function for the new Flower approach
def client_fn(cid: str):
    """Create client function for Flower simulation"""
    return IoTClient(client_id=int(cid))

if __name__ == "__main__":
    # For standalone client
    print("🚀 Starting IoT FL Client...")
    client = IoTClient(client_id=1)
    
    # Connect to SuperLink (new Flower approach)
    try:
        fl.client.start_client(
            server_address="127.0.0.1:9092",  # New default port
            client=client
        )
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure SuperLink is running: flower-superlink --insecure")
