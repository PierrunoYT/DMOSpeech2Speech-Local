# DMOSpeech2Speech

A text-to-speech synthesis system based on DMOSpeech2Speech with a user-friendly Gradio interface.

## Features

- **Zero-shot voice cloning** from reference audio
- **High-quality speech synthesis** with metric optimization
- **Easy-to-use Gradio interface** 
- **Configurable generation parameters**
- **Support for both CPU and GPU inference**

## Quick Start

**The fastest way to get started:**

1. **Download and setup:**
   ```bash
   # Create virtual environment first
   python -m venv dmo2
   
   # Activate environment
   # Linux/macOS:
   source dmo2/bin/activate
   # Windows:
   dmo2\Scripts\activate
   
   # Clone and enter project
   git clone https://github.com/yl4579/DMOSpeech2.git
   cd DMOSpeech2
   
   # IMPORTANT: Install PyTorch first (required for the interface)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Then install other requirements
   pip install -r DMOSpeech2/requirements.txt
   pip install ipython
   ```

2. **Download models:**
   ```bash
   mkdir ckpts
   cd ckpts
   
   # Download from Huggingface
   # Linux/macOS:
   wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt
   wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt
   
   # Windows (PowerShell):
   # Invoke-WebRequest -Uri "https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt" -OutFile "model_85000.pt"
   # Invoke-WebRequest -Uri "https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt" -OutFile "model_1500.pt"
   
   cd ..
   ```

3. **Launch interface:**
   ```bash
   python run_tts.py
   ```

4. **Open your browser** and start generating speech!

**For more advanced usage and inference examples, see [demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb)**

## Prerequisites

### Python Installation

Make sure you have Python 3.10+ installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Create Virtual Environment

**Recommended: Python 3.10**

**All platforms:**
```bash
python -m venv dmo2
```

**Activate the environment:**

**Linux/macOS:**
```bash
source dmo2/bin/activate
```

**Windows (Command Prompt):**
```cmd
dmo2\Scripts\activate
```

**Windows (PowerShell):**
```powershell
dmo2\Scripts\Activate.ps1
```

**Note:** You'll need to activate this environment every time you want to use DMOSpeech2Speech.

### Install Required Packages

1. Install PyTorch with CUDA support:

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Check your CUDA version:**
```bash
nvidia-smi
```

2. Install python requirements:
```bash
pip install -r DMOSpeech2/requirements.txt
```

3. **Additional dependencies:**
```bash
pip install ipython
```

**Alternative:** You can also create an [F5-TTS environment](https://github.com/SWivid/F5-TTS) and directly run the inference with it.

## Download Model Checkpoints

1. Create the ckpts folder:
```bash
mkdir ckpts
```

2. Download the model files from [Huggingface](https://huggingface.co/yl4579/DMOSpeech2) and place them in the `ckpts` folder:

```bash
mkdir ckpts
cd ckpts
wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt
wget https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt
cd ..
```

**Model descriptions:**
- `model_85000.pt` - DMOSpeech checkpoint (including teacher for teacher-guided sampling)
- `model_1500.pt` - GRPO-finetuned duration predictor checkpoint

**Expected folder structure:**
```
DMOSpeech2Speech/
├── README.md
├── run_tts.py
├── dmo_tts_interface.py
├── ckpts/
│   ├── model_85000.pt
│   └── model_1500.pt
├── DMOSpeech2/
│   ├── requirements.txt
│   └── src/
└── ...
```

## Running the Interface

**Option 1: Easy launcher (Recommended)**
```bash
python run_tts.py
```

**Option 2: Direct launch**
```bash
python dmo_tts_interface.py
```

The interface will open in your browser at localhost:7861

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'torch'" Error:**
- **First, verify your environment is activated:** 
  - Linux/macOS: `source dmo2/bin/activate`
  - Windows: `dmo2\Scripts\activate`
- **Check if PyTorch is actually installed:** `pip list | grep torch` (Linux/Mac) or `pip list | findstr torch` (Windows)
- **Test PyTorch import directly:** `python -c "import torch; print(torch.__version__)"`
- **If PyTorch is installed but still getting import error:**
  - Check which Python you're using: `which python` (Linux/Mac) or `where python` (Windows)
  - Make sure you're using the virtual environment Python
  - Try running with explicit Python: `python.exe dmo_tts_interface.py` (Windows)
  - Restart your terminal/command prompt and reactivate environment
- **If PyTorch is not installed:**
  - For CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
  - For CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  - For CPU-only: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- **Environment conflicts:**
  - Deactivate and reactivate environment: `deactivate` then reactivate
  - Try creating a fresh environment if issues persist

**Interface won't start:**
- Make sure you're in the project root directory (where `README.md` is located)
- Check that you have the `ckpts/` folder with model files
- Verify all dependencies are installed: `pip install -r DMOSpeech2/requirements.txt`
- Install additional dependencies: `pip install ipython`

**Model loading errors:**
- Ensure your checkpoint files are in the `ckpts/` directory
- The interface supports both EMA and non-EMA model formats

**Import errors:**
- Run from the project root directory: `cd /path/to/DMOSpeech2Speech`
- Make sure the `DMOSpeech2/src/` directory exists and contains the source files

**Audio generation issues:**
- Provide clear reference audio (3-10 seconds recommended)
- Ensure reference text matches the reference audio content
- Try adjusting generation parameters (guidance scale, steps)
- Check that your GPU has sufficient memory

**Port conflicts:**
- Interface runs on port 7861
- If port is busy, modify the port number in `dmo_tts_interface.py`
- Use `netstat -an | grep :7861` to check port availability

### Dependency Issues

**If you encounter import errors:**
- Make sure you have all Gradio dependencies: `pip install gradio>=3.45.2`
- Ensure you're running from the project root directory
- Check Python path issues by running: `python -c "import sys; print(sys.path)"`

**For Windows users:**
- If activation doesn't work in PowerShell, try Command Prompt instead
- Make sure you have execution policies set correctly for PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Use forward slashes or escape backslashes in paths when needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

Based on the original [DMOSpeech2](https://github.com/yl4579/DMOSpeech2) research by Yinghao Aaron Li et al.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.