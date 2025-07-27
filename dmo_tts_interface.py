#!/usr/bin/env python3
"""
DMOSpeech2 TTS Interface
A Gradio interface for DMOSpeech2 inference using the project's own models
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the DMOSpeech2 source to Python path
sys.path.insert(0, str(Path(__file__).parent / "DMOSpeech2" / "src"))

import gradio as gr
import torch
import numpy as np
import soundfile as sf
import torchaudio

try:
    from infer import DMOInference
except ImportError as e:
    print(f"Error importing DMOInference: {e}")
    print("Make sure you're running from the correct directory and have the DMOSpeech2 source available")
    sys.exit(1)

# Global inference model
dmo_model = None

def initialize_model():
    """Initialize the DMOSpeech2 model"""
    global dmo_model
    
    print("Initializing DMOSpeech2 model...")
    
    # Find required checkpoints
    ckpts_dir = Path("ckpts")
    if not ckpts_dir.exists():
        print("ckpts/ directory not found")
        return False
    
    student_checkpoint = ckpts_dir / "model_85000.pt"
    duration_checkpoint = ckpts_dir / "model_1500.pt"
    
    if not student_checkpoint.exists():
        print(f"Student checkpoint not found: {student_checkpoint}")
        return False
    
    if not duration_checkpoint.exists():
        print(f"Duration predictor checkpoint not found: {duration_checkpoint}")
        return False
    
    print(f"Using student checkpoint: {student_checkpoint}")
    print(f"Using duration predictor checkpoint: {duration_checkpoint}")
    
    try:
        # Initialize DMO inference model
        dmo_model = DMOInference(
            student_checkpoint_path=str(student_checkpoint),
            duration_predictor_path=str(duration_checkpoint),
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_type="F5TTS_Base",
            tokenizer="pinyin",
            dataset_name="Emilia_ZH_EN"
        )
        
        print("DMOSpeech2 model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def generate_speech(
    ref_audio_file,
    ref_text,
    gen_text,
    speed=1.0,
    guidance_scale=2.0,
    num_steps=4,
    seed=42
):
    """Generate speech using DMOSpeech2"""
    
    if dmo_model is None:
        return None, "Model not initialized. Please check that model checkpoints are available."
    
    if not ref_audio_file:
        return None, "Please provide reference audio."
    
    if not gen_text.strip():
        return None, "Please enter text to generate."
    
    try:
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # If no reference text provided, let DMOInference handle transcription
        prompt_text = ref_text if ref_text.strip() else None
        
        # Generate speech using DMOSpeech2
        generated_audio = dmo_model.generate(
            gen_text=gen_text,
            audio_path=ref_audio_file,
            prompt_text=prompt_text,
            teacher_steps=max(1, num_steps * 4),  # Scale up steps for teacher
            student_start_step=1,
            cfg_strength=guidance_scale,
            verbose=True
        )
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        # Save the generated audio
        sf.write(output_path, generated_audio, dmo_model.target_sample_rate)
        
        return output_path, f"Speech generated successfully! Sample rate: {dmo_model.target_sample_rate}Hz"
        
    except Exception as e:
        return None, f"Error generating speech: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="DMOSpeech 2 Speech", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # DMOSpeech 2 Speech
        
        A Gradio interface for DMOSpeech2 speech synthesis. Upload a reference audio file and enter the text you want to generate.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                ref_audio = gr.Audio(
                    label="Reference Audio (upload 3-10 seconds)",
                    type="filepath"
                )
                
                ref_text = gr.Textbox(
                    label="Reference Text (required)",
                    placeholder="Enter the text spoken in the reference audio",
                    lines=2
                )
                
                gen_text = gr.Textbox(
                    label="Text to Generate",
                    placeholder="Enter the text you want to synthesize...",
                    lines=4
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.3,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.5,
                        maximum=5.0,
                        value=2.0,
                        step=0.1
                    )
                    
                    num_steps = gr.Slider(
                        label="Generation Steps",
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=42,
                        precision=0
                    )
                
                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                output_audio = gr.Audio(
                    label="Generated Speech",
                    type="filepath"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                ref_audio,
                ref_text,
                gen_text,
                speed,
                guidance_scale,
                num_steps,
                seed
            ],
            outputs=[output_audio, status_text]
        )
        
        # Examples and tips
        gr.Markdown("""
        ### Tips:
        - Use clear, high-quality reference audio (3-10 seconds is ideal)
        - The reference text must accurately match what's spoken in the reference audio
        - DMOSpeech2 works best with Chinese and English text
        - Higher guidance scale makes output more similar to reference voice
        - More generation steps generally improve quality but take longer
        """)
    
    return interface

def main():
    """Main function to run the interface"""
    print("Starting DMOSpeech2 TTS Interface...")
    
    # Check if we're in the right directory
    if not Path("DMOSpeech2").exists():
        print("Error: DMOSpeech2 directory not found!")
        print("Please run this script from the root directory containing DMOSpeech2/")
        return
    
    # Initialize model
    if not initialize_model():
        print("Failed to initialize DMOSpeech2 model. Please check your setup.")
        return
    
    # Create and launch interface
    interface = create_interface()
    
    print("Launching interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()