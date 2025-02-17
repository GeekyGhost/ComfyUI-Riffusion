import sys
import subprocess
from pathlib import Path

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
print("All packages needed by Riffusion are installed.")

### Don't import nodes until all packages are installed ###
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']