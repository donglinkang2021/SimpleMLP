<div align="center">

# Simple MLP

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CPU](https://img.shields.io/badge/CPU-x86__64-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243)
![Plotly](https://img.shields.io/badge/Plotly-5.19.0-3F4F75)

</div>

> Inspired by [SimpleAttention](https://github.com/donglinkang2021/SimpleAttention).

Just use simple MLP to regress or classify.

## Run it💨

You can run the code with the following command:

```bash
python run.py --multirun dataset=regress_plane,regress_gaussian,classify_two_gauss,classify_spiral,classify_circle,classify_xor model=mlp_relu_1h,mlp_relu_2h,mlp_tanh_1h,mlp_tanh_2h,mlp_silu_1h,mlp_silu_2h,feat_attn_1h,feat_attn_2h,feat_attn_3h optimizer=adam,sgd
```

> On my machine(just a laptop with AMD Ryzen 7 5800H and 14GB RAM), it takes about 2 minutes to run all the settings under the combinations of 6 datasets, 9 models and 2 optimizers. (108 experiments in total)

## Visualization📊

I deploy two visualization apps on streamlit cloud, you can check it out here:

Or you can run the apps locally:

```bash
streamlit run app_datasets.py # for datasets visualization, just 2D scatter plot
streamlit run app_results.py # for training results visualization, including `config.yaml` and loss curve of 108 experiments
```