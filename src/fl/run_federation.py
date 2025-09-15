import subprocess
import time
import threading
import sys
import os

def run_server(num_rounds=3):
    """Run FL server in separate process"""
    print("🚀 Starting FL Server...")
    
    try:
        server_process = subprocess.Popen([
            sys.executable, "src/fl/flower_server.py", 
            "--rounds", str(num_rounds)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give server time to start
        time.sleep(5)
        
        return server_process
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def run_client(client_id, dataset="n_baiot", server_address="localhost:8080"):
    """Run FL client"""
    print(f"📱 Starting Client {client_id} with {dataset}...")
    
    try:
        # Import here to avoid circular imports
        from flower_client import create_client
        
        client = create_client(client_id, dataset)
        
        # Connect to server
        fl.client.start_numpy_client(server_address=server_address, client=client)
        
    except Exception as e:
        print(f"❌ Client {client_id} failed: {e}")

def simulate_federated_learning(num_clients=3, num_rounds=3):
    """Simulate complete FL setup"""
    print("🌐 Simulating Federated Learning Setup")
    print("=" * 50)
    
    # Start server
    server_process = run_server(num_rounds)
    
    if server_process is None:
        print("❌ Cannot start federated learning without server")
        return
    
    # Start clients with different datasets
    datasets = ['n_baiot', 'bot_iot', 'ton_iot']
    client_threads = []
    
    for i in range(num_clients):
        dataset = datasets[i % len(datasets)]
        
        def run_client_thread(client_id=i, ds=dataset):
            try:
                run_client(client_id, ds)
            except Exception as e:
                print(f"Client {client_id} error: {e}")
        
        thread = threading.Thread(target=run_client_thread)
        thread.daemon = True
        client_threads.append(thread)
        thread.start()
        
        # Stagger client starts
        time.sleep(2)
    
    print(f"✅ Started {num_clients} clients")
    print("🔄 Federated learning in progress...")
    
    # Wait for completion
    try:
        server_process.wait(timeout=60)  # Wait max 60 seconds
    except subprocess.TimeoutExpired:
        print("⏰ FL training timeout")
        server_process.terminate()
    
    # Wait for threads to finish
    for thread in client_threads:
        thread.join(timeout=5)
    
    print("✅ Federated Learning Simulation Complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    
    args = parser.parse_args()
    
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Need to install flower first
    try:
        import flwr as fl
        simulate_federated_learning(args.clients, args.rounds)
    except ImportError:
        print("❌ Flower not installed. Install with:")
        print("pip install flwr[simulation]")
