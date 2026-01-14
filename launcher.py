import uvicorn
import webbrowser
import threading
import time
import os
import sys
import socket
import subprocess
import shutil

def is_port_open(host, port):
    """Checks if a service is running on a specific port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def start_ollama():
    """Attempts to start Ollama if it's not running."""
    print(">> Ollama is not running. Attempting to start...")
    
    # Check if ollama is installed
    if not shutil.which("ollama"):
        print("!! ERROR: 'ollama' command not found.")
        print("!! Please install Ollama from https://ollama.com")
        return False

    try:
        # Start Ollama in a separate process
        # stdout=subprocess.DEVNULL hides the logs to keep console clean
        if os.name == 'nt': # Windows
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else: # Mac/Linux
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print(">> Waiting for Ollama to initialize...")
        
        # Wait up to 10 seconds for it to start
        for _ in range(10):
            if is_port_open("127.0.0.1", 11434):
                print(">> Ollama started successfully!")
                return True
            time.sleep(1)
            
        print("!! WARNING: Ollama started but port 11434 is not responding yet.")
        return True
        
    except Exception as e:
        print(f"!! Failed to start Ollama: {e}")
        return False

def open_browser():
    """Waits for the server to start, then opens the browser."""
    time.sleep(2) 
    print(">> Opening Browser to http://127.0.0.1:8000")
    webbrowser.open("http://127.0.0.1:8000")

def main():
    # 1. Check for main application file
    if not os.path.exists("main.py"):
        print("ERROR: 'main.py' not found!")
        input("Press Enter to exit...")
        sys.exit(1)

    # 2. Check/Start Ollama
    if not is_port_open("127.0.0.1", 11434):
        start_ollama()
    else:
        print(">> Ollama is already running.")

    # 3. Schedule the browser to open
    threading.Thread(target=open_browser, daemon=True).start()

    # 4. Start the Server
    print(">> Starting AI Prompt Director...")
    print(">> Press Ctrl+C to stop.")
    
    try:
        uvicorn.run(
            "main:app", 
            host="127.0.0.1", 
            port=8000, 
            reload=False, # Disable reload for stability in launcher mode
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n>> Stopping...")

if __name__ == "__main__":
    main()