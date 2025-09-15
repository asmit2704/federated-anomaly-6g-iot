import flwr as fl
from flower_client import client_fn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_simulation():
    """Run FL simulation with multiple clients"""
    print("🌐 Starting Federated Learning Simulation...")
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,          # All clients participate  
        fraction_evaluate=1.0,     # All clients evaluate
        min_fit_clients=2,         # Minimum 2 clients
        min_evaluate_clients=2,    # Minimum 2 for evaluation
        min_available_clients=2,   # Wait for 2 clients
    )
    
    # Run simulation
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,           # Client creation function
            num_clients=3,                 # Total number of clients
            config=fl.server.ServerConfig(num_rounds=3),  # 3 FL rounds
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0}  # Resource per client
        )
        
        print("\n📊 FL Simulation Results:")
        print(f"✅ Completed {len(history.losses_distributed)} rounds")
        
        # Show final metrics
        if history.metrics_distributed:
            final_accuracy = history.metrics_distributed[-1][1]['accuracy']
            print(f"✅ Final accuracy: {final_accuracy:.4f}")
        
        print("🎉 Federated Learning Simulation Complete!")
        
        return history
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return None

if __name__ == "__main__":
    run_simulation()
