"""Create a local venv even when Ubuntu's optional ensurepip package is absent."""
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
import venv

target = pathlib.Path(sys.argv[1]).resolve()
venv.EnvBuilder(with_pip=False).create(target)
python = target/'bin/python'
check = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True)
if check.returncode:
    with tempfile.TemporaryDirectory(prefix='deform-pip-') as temp:
        bootstrap = pathlib.Path(temp)/'get-pip.py'
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', bootstrap)
        subprocess.run([str(python), str(bootstrap)], check=True)
