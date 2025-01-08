import subprocess

if __name__ == "__main__":
    processes = []
    # Start a process for each PID
    for pid in range(4):
        # Define the command and arguments
        command = ['python', 'ffhq_fid.py', '--pid', str(pid)]
        
        # Start the process
        process = subprocess.Popen(command)

        processes.append(process)
    
    for process in processes:
        process.wait()

    print("All processes have finished.")

        