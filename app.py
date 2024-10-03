import streamlit as st
import plotly.graph_objs as go
import dataset
import numpy as np

# Set page title
st.set_page_config(page_title="Dataset Visualization Tool", page_icon="🔍")

# Dataset selection
dataset_options = {
    "Regression Plane": dataset.regress_plane,
    "Regression Gaussian": dataset.regress_gaussian,
    "Two Gaussian Classification": dataset.classify_two_gauss_data,
    "Spiral Classification": dataset.classify_spiral_data,
    "Circle Classification": dataset.classify_circle_data,
    "XOR Classification": dataset.classify_xor_data
}

dataset_name = st.sidebar.selectbox("Select Dataset", list(dataset_options.keys()))
selected_function = dataset_options[dataset_name]

# Parameter adjustment
st.sidebar.header("Parameter Settings")

num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=5000, value=1000, step=100)
noise = st.sidebar.slider("Noise Level", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
radius = st.sidebar.slider("Radius", min_value=1, max_value=10, value=6, step=1)

# Additional parameters
if dataset_name == "XOR Classification":
    padding = st.sidebar.slider("Padding", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    data = selected_function(num_samples, noise, radius, padding)
else:
    data = selected_function(num_samples, noise, radius)

x, y, label = data

# Create Plotly chart
if dataset_name in ["Regression Plane", "Regression Gaussian"]:
    # For regression tasks, use continuous color mapping
    colors = label
    colorscale_options = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']
    colorscale = st.sidebar.selectbox("Select Color Scheme", colorscale_options, index=colorscale_options.index('rdbu'))

else:
    # For classification tasks, use discrete color mapping
    color_map = {1: 'red', -1: 'blue'}
    colors = [color_map[int(l)] for l in label]
    colorscale = None

fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        color=colors,
        colorscale=colorscale,
        size=8,
        opacity=0.7
    )
))

# Add plot size adjustment slider
plot_size = st.sidebar.slider("Plot Size", min_value=300, max_value=1000, value=600, step=50)

fig.update_layout(
    title=dict(
        text=f"{dataset_name} Dataset Visualization",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    showlegend=False,
    width=plot_size,
    height=plot_size,
    autosize=False,
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)


st.plotly_chart(fig, use_container_width=False)