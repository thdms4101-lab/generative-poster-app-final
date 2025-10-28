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
    y = center[1] + radii * np.sin(angles)
    return x, y

# --- [Block 2]: Palette Generator (Modified for Streamlit) ---

def make_palette(k=6, mode="pastel", base_h=0.60, custom_palette=None):
    """
    Generates a list of k colors based on the selected mode.
    Replaces "csv" mode with "custom" (from session_state).
    """
    if mode == "custom":
        if not custom_palette:
            st.warning("Your custom palette is empty. Using 'pastel' instead.")
            mode = "pastel" # Fallback
        else:
            # Convert list of dicts to list of (r,g,b) tuples
            return [(c['r'], c['g'], c['b']) for c in custom_palette]

    cols = []
    for _ in range(k):
        if mode == "pastel":
            h = random.random(); s = random.uniform(0.15,0.35); v = random.uniform(0.9,1.0)
        elif mode == "vivid":
            h = random.random(); s = random.uniform(0.8,1.0);  v = random.uniform(0.8,1.0)
        elif mode == "mono":
            h = base_h;           s = random.uniform(0.2,0.6);  v = random.uniform(0.5,1.0)
        else: # random
            h = random.random(); s = random.uniform(0.3,1.0); v = random.uniform(0.5,1.0)
        cols.append(tuple(hsv_to_rgb([h,s,v])))
    return cols

# --- [Block 3]: Main Poster Drawing Function (Modified) ---

def draw_poster(n_layers, wobble, palette_mode, seed, shape, petals, 
                drawing_style, bg_color_rgb, custom_palette):
    """
    The main function that draws the poster.
    Does NOT call plt.show(). Returns fig.
    """
    random.seed(seed); np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis('off')
    ax.set_facecolor(bg_color_rgb) # Use color from UI

    palette = make_palette(6, mode=palette_mode, custom_palette=custom_palette)
    if not palette:
        st.error("Error: Palette is empty.")
        palette = [(0.1, 0.1, 0.1)] # Default fallback

    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)

        # --- Shape Selection ---
        if shape == "heart":
            x, y = heart((cx,cy), r=rr, wobble=wobble)
        elif shape == "flower":
            x, y = flower((cx,cy), r=rr, wobble=wobble, petals=petals)
        else: # 'blob'
            x, y = blob((cx,cy), r=rr, wobble=wobble)

        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.6)

        # --- Drawing Style Selection ---
        if drawing_style == "line":
            x_line = np.append(x, x[0])
            y_line = np.append(y, y[0])
            linewidth = random.uniform(1.0, 3.0)
            ax.plot(x_line, y_line, color=color, alpha=alpha, lw=linewidth)
        else: # 'fill'
            ax.fill(x, y, color=color, alpha=alpha, edgecolor='none')

    ax.text(0.05, 0.95, f"Final Poster • {shape.title()} / {palette_mode.title()}",
            transform=ax.transAxes, fontsize=12, weight="bold")
    
    return fig # Return the figure object

# --- [Block 4]: Streamlit App & UI ---

st.set_page_config(layout="wide")
st.title("Final Project: Interactive Poster Generator 🎨")

# --- Initialize Session State (Replaces palette.csv) ---
if 'palette' not in st.session_state:
    # Load the 9 blue colors from your example script as the default
    st.session_state.palette = [
        {"name":'powder_blue', "r":0.69, "g":0.87, "b":0.90},
        {"name":'light_blue',  "r":0.67, "g":0.84, "b":0.90},
        {"name":'deep_sky',    "r":0.0,  "g":0.74, "b":1.0},
        {"name":'dodger_blue', "r":0.11, "g":0.56, "b":1.0},
        {"name":'royal_blue',  "r":0.25, "g":0.41, "b":0.88},
        {"name":'medium_blue', "r":0.0,  "g":0.0,  "b":0.80},
        {"name":'dark_blue',   "r":0.0,  "g":0.0,  "b":0.54},
        {"name":'navy_blue',   "r":0.0,  "g":0.0,  "b":0.50},
        {"name":'midnight_blue',"r":0.09,"g":0.09, "b":0.43}
    ]

if 'seed' not in st.session_state:
    st.session_state.seed = random.randint(0, 10000)

# --- Sidebar UI (Replaces ipywidgets.interact) ---
st.sidebar.title("Controls")

# --- Poster Controls ---
st.sidebar.subheader("Poster Settings")
n_layers = st.sidebar.slider("Layers", 3, 25, 10)
wobble = st.sidebar.slider("Wobble", 0.01, 0.4, 0.15)
palette_mode = st.sidebar.selectbox("Color Mode", ["pastel","vivid","mono","random","custom"], index=4)
shape = st.sidebar.selectbox("Shape", ["blob","heart","flower"], index=0)

# Conditional slider: Only show "Petals" if shape is "flower"
petals = 7 # Default value
if shape == "flower":
    petals = st.sidebar.slider("Petals (꽃잎)", 3, 12, 7)

drawing_style = st.sidebar.selectbox("Draw Style", ["fill","line"], index=0)
bg_hex = st.sidebar.color_picker("Background Color", "#F7F7F7")
bg_rgb = to_rgb(bg_hex) # Convert hex to (0-1) RGB

# --- Seed Control ---
st.sidebar.subheader("Layout Seed")
if st.sidebar.button("New Random Layout"):
    st.session_state.seed = random.randint(0, 10000)
seed = st.sidebar.number_input("Current Seed", value=st.session_state.seed, step=1)
st.session_state.seed = seed

# --- Palette Manager UI (Replaces palette.csv logic) ---
st.sidebar.subheader("Custom Palette Manager")

# Display current palette
if st.session_state.palette:
    df = pd.DataFrame(st.session_state.palette)
    st.sidebar.dataframe(df)
else:
    st.sidebar.info("Your custom palette is empty.")

# --- Form to Add Color ---
with st.sidebar.form("add_color_form"):
    st.write("Add a new color:")
    new_name = st.text_input("Color Name", "new_color")
    new_hex = st.color_picker("Pick a color", "#FF0000")
    submitted = st.form_submit_button("Add Color")
    
    if submitted:
        new_rgb = to_rgb(new_hex)
        st.session_state.palette.append({
            "name": new_name,
            "r": new_rgb[0],
            "g": new_rgb[1],
            "b": new_rgb[2]
        })
        st.rerun()

# --- UI to Delete Color ---
st.sidebar.write("Delete a color:")
if st.session_state.palette:
    color_names = [c["name"] for c in st.session_state.palette]
    name_to_delete = st.sidebar.selectbox("Select color to delete", color_names)
    
    if st.sidebar.button("Delete Color"):
        st.session_state.palette = [c for c in st.session_state.palette if c["name"] != name_to_delete]
        st.rerun()
else:
    st.sidebar.text("No colors to delete.")


# --- Main Area: Draw the Poster ---
fig = draw_poster(
    n_layers=n_layers, 
    wobble=wobble, 
    palette_mode=palette_mode, 
    seed=seed, 
    shape=shape, 
    petals=petals, 
    drawing_style=drawing_style, 
    bg_color_rgb=bg_rgb, 
    custom_palette=st.session_state.palette
)

st.pyplot(fig)

# --- Download Button ---
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=bg_rgb)

st.sidebar.download_button(
    label="Download Poster (PNG)",
    data=buf.getvalue(),
    file_name=f"poster_{shape}_{palette_mode}_{seed}.png",
    mime="image/png"
)
