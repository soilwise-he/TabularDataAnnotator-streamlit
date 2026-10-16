import subprocess
result = subprocess.run([
    r'C:\Users\vercrume\.local\bin\uv.exe',
    'run',
    'python',
    '-m',
    'pytest',
    'tests/test_zenodo_autopilot.py',
    '-q'
], cwd=r'C:\Users\vercrume\Documents\LocalRepos\DataAnnotator', capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print('RETURNCODE', result.returncode)
