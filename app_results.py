import streamlit as st
import plotly.graph_objs as go
import os
import yaml
import json
import pandas as pd

# Set page title
st.set_page_config(page_title="Results Visualization Tool", page_icon="🔍")

# Display results from the result folder
st.sidebar.header("Results")

# Dataset selection
dataset_options = {
    "Regression Plane": "regress_plane",
    "Regression Gaussian": "regress_gaussian",
    "Two Gaussian Classification": "classify_two_gauss",
    "Spiral Classification": "classify_spiral",
    "Circle Classification": "classify_circle",
    "XOR Classification": "classify_xor"
}

model_options = {
    "Feature Attention 1 Hidden Layer": "feat_attn_1h",
    "Feature Attention 2 Hidden Layer": "feat_attn_2h",
    "Feature Attention 3 Hidden Layer": "feat_attn_3h",
    "MLP with 1 Hidden Layer, use tanh": "mlp_tanh_1h",
    "MLP with 2 Hidden Layer, use tanh": "mlp_tanh_2h",
    "MLP with 1 Hidden Layer, use ReLU": "mlp_relu_1h",
    "MLP with 2 Hidden Layer, use ReLU": "mlp_relu_2h",
    "MLP with 1 Hidden Layer, use SiLU": "mlp_silu_1h",
    "MLP with 2 Hidden Layer, use SiLU": "mlp_silu_2h"
}

optimizer_options = {
    "Adam": "adam",
    "SGD": "sgd"
}

dataset_name = st.sidebar.selectbox("Select Dataset", list(dataset_options.keys()))
model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
optimizer_name = st.sidebar.selectbox("Select Optimizer", list(optimizer_options.keys()))
selected_model = model_options[model_name]
selected_function = dataset_options[dataset_name]
selected_optimizer = optimizer_options[optimizer_name]
result_dir = f"result/{selected_function}_{selected_model}_{selected_optimizer}"

# Plot metrics
metrics_file = "metrics.json"
st.subheader(f"Metrics in {metrics_file}")
with open(os.path.join(result_dir, metrics_file), 'r') as file:
    metrics = json.load(file)
metrics_df = pd.DataFrame(metrics)
fig_results = go.Figure()
for col in metrics_df.columns[:-1]:
    fig_results.add_trace(
        go.Scatter(x=metrics_df.index, y=metrics_df[col], mode='lines', name=col)
    )
fig_results.update_layout(
    title=dict(
        text=f"Training Metrics",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Step",
    yaxis_title="Loss",
    showlegend=True
)
st.plotly_chart(fig_results, use_container_width=True)
    
# Display parameters
param_file = "config.yaml"
with open(os.path.join(result_dir, param_file), 'r') as file:
    params = yaml.safe_load(file)
    st.subheader(f"Parameters in {param_file}")
    st.write(params)