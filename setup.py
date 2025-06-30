# setup.py
from setuptools import setup, find_packages

setup(
    name="bci-processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.2.0',
        'scipy>=1.10.0',
        'pywavelets>=1.4.0',
        'pyyaml>=6.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for EEG signal processing and classification for BCI applications",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bci-processor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)