import io
import typing as T
import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
import torch
import torchaudio
from pedalboard.io import AudioFile
import glob
import wave

MAX_BATCH_SIZE = 8

class RiffusionNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                'spectrogram': ("IMAGE",),
                'output_type': ("STRING", {"default": "audio/wav"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, spectrogram, output_type):
        # Your processing logic here
        pass

NODE_CLASS_MAPPINGS = {
    "RiffusionNode": RiffusionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RiffusionNode": "Riffusion"
}