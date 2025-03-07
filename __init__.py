import sys
import subprocess
from pathlib import Path

FFMPEG_IS_NOT_INSTALLED = False

### GET PACKAGES ###
def packages(versions=False):
    try:
        result = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.STDOUT)
        lines = result.decode().splitlines()
        return [line if versions else line.split('==')[0] for line in lines]
    except subprocess.CalledProcessError as e:
        print("An error occurred while fetching packages:", e.output.decode())
        return []

### INSTALL PACKAGE ###
def install_package(package):
    print(f"Installing package {package}...")
    subprocess.check_call([sys.executable, '-s', '-m', 'pip', '-q', 'install', package])

### READ REQUIREMENTS ###
def read_requirements():
    req_file = Path(__file__).parent / 'requirements.txt'
    if not req_file.exists():
        return []
    with open(req_file, 'r') as f:
        return [line.strip().split('>=')[0] for line in f if line.strip() and not line.startswith('#')]

### Ensure all required packages are installed ###
installed_packages = packages()
required_packages = read_requirements()

for package in required_packages:
    if package.lower() not in [pkg.lower() for pkg in installed_packages]:
        install_package(package)
print("All Python packages needed by Riffusion are installed.")

### Ensure FFmpeg is installed ###
# FFmpeg is a 3rd-party program that is needed for Riffusion to work and it's not a Python library.
# determine if system is windows or linux??? MacOS is not really supported for Stable Diffusion
try:
    result = subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
    print("FFmpeg is installed.")
except subprocess.CalledProcessError as e:
    print("[Riffusion WARNING] An error occurred while checking FFmpeg installation:", e.output.decode())
    print("[Riffusion WARNING] All other audio formats will be disabled until until the previous error is resolved.")
    FFMPEG_IS_NOT_INSTALLED = True
except FileNotFoundError: #Windows gives a FileNotFoundError
    print("[Riffusion WARNING] FFmpeg is not installed.")
    print("[Riffusion WARNING] Please install FFmpeg from https://ffmpeg.org/download.html. All other audio formats will be disabled until FFmpeg is installed.")
    # set flag to signal FFMpeg is not installed
    FFMPEG_IS_NOT_INSTALLED = True

# TODO: Add FFmpeg installer for Windows
# TODO: Add FFmpeg installer for Linux

### Don't import nodes until all packages are installed ###
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']