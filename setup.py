from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent
REQUIREMENTS = [i.strip() for i in open(root / 'requirements.txt').readlines()]

setup(name='tf_utils',
      version='0.1',
      description='some usefull things for tensorflow',
      install_requires=REQUIREMENTS,
      packages=find_packages())

