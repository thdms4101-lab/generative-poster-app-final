import streamlit as st
import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_rgb, to_hex
from io import BytesIO

# --- [Block 1]: Shape Functions (Unchanged) ---

def blob(center=(0.5, 0.5), r=0.3, points=200, wobble=0.15):
    """Generate a wobbly closed shape."""
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    radii  = r * (1 + wobble*(np.random.rand(points)-0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

def heart(center=(0.5, 0.5), r=0.3, points=200, wobble=0.15):
    """Generate coordinates for a wobbly heart shape."""
    t = np.linspace(0, 2*math.pi, points, endpoint=False)
    base_x = 16 * np.sin(t)**3
    base_y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    x_norm = base_x / 16.0
    y_norm = base_y / 16.0
    wobble_factor = 1 + wobble*(np.random.rand(points)-0.5)
    x = center[0] + x_norm * r * wobble_factor
    y = center[1] + y_norm * r * wobble_factor
    return x, y

def flower(center=(0.5, 0.5), r=0.3, points=200, wobble=0.1, petals=7):
    """Generate coordinates for a wobbly flower shape."""
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    base_radii = r * (1 + 0.3 * np.sin(angles * petals))
    wobble_factor = 1 + wobble*(np.random.rand(points)-0.5)
    radii = base_radii * wobble_factor
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii
