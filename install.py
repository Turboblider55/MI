import os
import subprocess
import sys

def create_virtualenv(env_path):
    subprocess.run([sys.executable, '-m', 'venv', env_path])

def install_dependencies(env_path):
    pip_path = os.path.join(env_path, 'Scripts', 'pip') if os.name == 'nt' else os.path.join(env_path, 'bin', 'pip')
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'])

def activate_virtualenv(env_path):
    activate_script = os.path.join(env_path, 'Scripts', 'activate.bat') if os.name == 'nt' else os.path.join(env_path, 'bin', 'activate')
    command = f'"{activate_script}" && your_next_command'
    subprocess.run(command, shell=True, executable='/bin/bash' if os.name != 'nt' else None)

if __name__ == '__main__':
    env_path = input('Adja meg a környezet elérési útját: ')
    create_virtualenv(env_path)
    activate_virtualenv(env_path)
    install_dependencies(env_path)
    
