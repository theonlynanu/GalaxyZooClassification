"""
Danyal Ahmed - April 2026

models/
Package for model architectures used across the GZ2 projects.

Each file is a self-contained model, this file allows you to use:
    
    from models import BasicCNN
    
New models can be added to this directory and imported by adding them to __all__ 
below.
"""
from models.basic_cnn import BasicCNN

__all__ = ["BasicCNN"]
