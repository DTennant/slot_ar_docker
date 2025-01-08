import torch
import time

def occupy_gpu_memory_and_compute():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # List to store the tensors
    tensors = []
    
    for i in range(num_gpus):
        # Set the current GPU
        torch.cuda.set_device(i)
        
        # Get the total memory of the current GPU
        total_memory = torch.cuda.get_device_properties(i).total_memory
        
        # Calculate 90% of the total memory (in bytes)
        memory_to_occupy = int(total_memory * 0.9)
        
        # Create a tensor that occupies 90% of the GPU memory
        tensor = torch.empty(memory_to_occupy, dtype=torch.uint8, device=f'cuda:{i}')
        
        # Perform a small operation to ensure the memory is allocated
        tensor += 1
        
        # Keep a reference to the tensor
        tensors.append(tensor)
        
        print(f"Occupied GPU {i} memory")
    
    print("Press Ctrl+C to stop and release GPU resources.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping and releasing GPU resources...")

if __name__ == "__main__":
    occupy_gpu_memory_and_compute()