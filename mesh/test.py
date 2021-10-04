import subprocess

cl_paths = subprocess.check_output(['where', 'cl']).decode().split('\r\n')
print(cl_paths)