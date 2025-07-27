#!/usr/bin/env python3
"""
Simple launcher for the TTS interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the TTS interface"""
    
    # Check if we're in the right directory
    if not Path("DMOSpeech2").exists():
        print("Error: DMOSpeech2 directory not found!")
        print("Please run this script from the root directory containing DMOSpeech2/")
        return 1
    
    # Check if the DMO interface exists
    if Path("dmo_tts_interface.py").exists():
        interface_script = "dmo_tts_interface.py"
        port = 7861
        print("Starting DMOSpeech2 TTS Interface...")
    else:
        print("Error: DMOSpeech2 TTS interface not found!")
        print("Expected dmo_tts_interface.py in the current directory")
        return 1
    
    print(f"This will open in your web browser at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the interface
        subprocess.run([sys.executable, interface_script], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running interface: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())