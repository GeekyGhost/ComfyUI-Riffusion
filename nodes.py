import io
import typing as T
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
import torch
import torchaudio
from pedalboard.io import AudioFile
import glob
import wave
import platform

MAX_BATCH_SIZE = 8

class RiffusionNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                'spectrogram': ("IMAGE",),
                'filename_prefix': ("STRING", {"default": "audio\Riffusion"}),

            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "Process_Riffusion"
    CATEGORY = "audio"
    OUTPUT_NODE = True

    def spectrogram_from_image(self, image, max_volume=50, power_for_image=0.25) -> np.ndarray:
        # Convert the image to a numpy array
        data = np.array(image).astype(np.float32)
        # Reverse the image vertically
        data = data[::-1,:,0]
        # Invert the image  
        data = 255-data
        # Normalize the image
        data = data*max_volume/255
        # Invert power curve
        data = np.power(data, 1/power_for_image)
        return data

    def waveform_from_spectrogram(self, Sxx: np.ndarray, n_fft: int, hop_length: int, win_length: int, num_samples: int, 
        sample_rate: int, mel_scale: bool = True, nmels: int = 512, max_mel_iters: int = 200, num_griffin_lim_iters: int = 32,
        device: str = platform.system() == "Darwin" and "cpu" or "cuda"
    ) -> np.ndarray:
        Sxx_torch = torch.from_numpy(Sxx).to(device)
        
        if mel_scale:
            mel_inv_scaler = torchaudio.transforms.InverseMelScale(n_mels=nmels, sample_rate=sample_rate, f_min=0, f_max=10000,
                n_stft=n_fft // 2 + 1, norm=None, mel_scale="htk").to(device)
            Sxx_torch = mel_inv_scaler(Sxx_torch)
        
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1.0, 
            n_iter=num_griffin_lim_iters).to(device)

        waveform = griffin_lim(Sxx_torch).cpu().numpy()

        return waveform


    def get_wave_bytes_from_spectrogram(self, spec) -> T.Tuple[io.BytesIO, float]:
        max_volume = 50
        power_for_image = 0.25
        Sxx = self.spectrogram_from_image(spec, max_volume, power_for_image)
        
        sample_rate = 44100
        clip_duration_ms = 5000

        bins_per_image = spec.height
        n_mels = spec.height

        #FFT parameters
        window_duration_ms = 100
        padded_duration_ms = 400
        step_size_ms = 10

        #Derived Parameters
        num_samples = (
            int(spec.width/float(bins_per_image) * clip_duration_ms) * sample_rate
        )
        n_fft = int(padded_duration_ms / 1000 * sample_rate)
        hop_length = int(step_size_ms / 1000 * sample_rate)
        win_length = int(window_duration_ms / 1000 * sample_rate)

        #Generate the waveform
        samples = self.waveform_from_spectrogram(Sxx=Sxx, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
            num_samples=num_samples, sample_rate=sample_rate, mel_scale=True, nmels=n_mels, max_mel_iters=200, 
            num_griffin_lim_iters=32)

        #Encode samples and return to calling function
        wav_bytes = io.BytesIO()
        wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
        wav_bytes.seek(0)

        duration_s = float(len(samples))/sample_rate

        return wav_bytes, duration_s

    def process_wav(self, wav_file):
        with AudioFile(wav_file) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        filename = wav_file.replace(".wav", ".mp3")

        with AudioFile(filename, "w", sample_rate, audio.shape[0]) as f:
            f.write(audio)
        

    def Process_Riffusion(self, spectrogram, filename_prefix):
        # Get the current file's directory and go up two levels
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent.parent / 'output'
        results = list()
        
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
         
        # Search output_dir for files with the filename prefix
        list_of_files = glob.glob(str(output_dir / f"{filename_prefix}*"))

        # Get the highest number in the list of files
        if not list_of_files:
            new_number = 0
        else:
            max_number = 0
            for file in list_of_files:
                number = int(file.split("_")[-2]) 
                max_number = max(max_number, number)
            # Increment the number and add it to the filename and add the extension
            new_number = max_number + 1

        new_filename = f"{filename_prefix}_{new_number:05}_.wav"

        # now load the spectrogram and convert it to a audio file
        # TODO: Add support for multiple images (as input) to generate a single audio file for longer audio lengths
        # TODO: Add support for multiple images to generate multiple audio files
        MAX_FILES = 1
        spec = None
        for (batch_number, image) in enumerate(spectrogram):
            print("Batch Number ", batch_number)
            if batch_number >= MAX_FILES:
                break
            i = 255. * image.cpu().numpy()
            spec = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        audio, duration = self.get_wave_bytes_from_spectrogram(spec)
        print(f"Duration of {new_filename}: {duration}")

        # get full output path
        full_output_path = os.path.join(output_dir, new_filename)

        # save the audio
        with open(full_output_path, 'wb') as f:
            f.write(audio.getbuffer())

        # TODO: add support for other audio types (wav, flac, mp3, ogg, aac)
        # deprecated until I figure out how to save as other audio types
        # Audio file formats to use:
        #   .wav
        #   .flac
        #   .mp3
        #   .ogg
        #   .aac
        #self.process_wav(full_output_path)

        results.append({
            "filename": new_filename,
            "subfolder": output_dir,
            "type": "output"
        })

        return { "ui": { "audio": results } }
        


NODE_CLASS_MAPPINGS = {
    "RiffusionNode": RiffusionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RiffusionNode": "Riffusion"
}