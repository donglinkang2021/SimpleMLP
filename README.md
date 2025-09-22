<div align="center">

# Simple MLP

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CPU](https://img.shields.io/badge/CPU-x86__64-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243)
![Plotly](https://img.shields.io/badge/Plotly-5.19.0-3F4F75)
![Hydra](https://img.shields.io/badge/HYDRA-1.3.2-89B8CD)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C)

</div>

> Inspired by [SimpleAttention](https://github.com/donglinkang2021/SimpleAttention).

Just use simple MLP to regress or classify.

## Run it💨

For the environment setup, we recommend using `uv` (super fast):

```bash
uv sync
```

You can run the code with the following command:

```bash
bash run.sh
```

## Data Visualization📊

I deploy a dataset visualization app on streamlit cloud, you can check it out here ➡ [Datasets Visualization](https://donglinkang2021-simplemlp-app-datasets-t3w3g8.streamlit.app/)

Or you can run the apps locally:

```bash
streamlit run app_datasets.py # for datasets visualization, just 2D scatter plot
```

## Evaluation🧪

> I evaluate the models based on the following metrics:

$$
\mathcal{score} = 0.4 \times \mathcal{train\_rank} + 0.3 \times \mathcal{val\_rank} + 0.3 \times \mathcal{test\_rank}
$$

The details can be found in the notebook `notebook\analyze.ipynb`.

- Rank(without weight initialization)

|              |   train |   val |   test |   score |   rank |
|:-------------|--------:|------:|-------:|--------:|-------:|
| Feat_Attn_3h |       3 |   1.5 |    2   |    2.25 |      1 |
| Feat_Attn_2h |       4 |   3   |    1   |    2.8  |      2 |
| Feat_Attn_1h |       1 |   4.5 |    4.5 |    3.1  |      3 |
| MLP_Relu_2h  |       5 |   1.5 |    4.5 |    3.8  |      4 |
| MLP_Silu_2h  |       2 |   8   |    9   |    5.9  |      5 |
| MLP_Tanh_2h  |       6 |   4.5 |    8   |    6.15 |      6 |
| MLP_Relu_1h  |       8 |   7   |    3   |    6.2  |      7 |
| MLP_Silu_1h  |       7 |   9   |    6   |    7.3  |      8 |
| MLP_Tanh_1h  |       9 |   6   |    7   |    7.5  |      9 |

- Rank(with weight initialization)

|              |   train |   val |   test |   score |   rank |
|:-------------|--------:|------:|-------:|--------:|-------:|
| MLP_Relu_2h  |     2   |     3 |      3 |     2.6 |      1 |
| MLP_Relu_1h  |     1   |     6 |      2 |     2.8 |      2 |
| MLP_Tanh_2h  |     7   |     1 |      1 |     3.4 |      3 |
| Feat_Attn_1h |     3.5 |     4 |      5 |     4.1 |      4 |
| MLP_Silu_1h  |     5   |     2 |      7 |     4.7 |      5 |
| MLP_Tanh_1h  |     3.5 |     8 |      6 |     5.6 |      6 |
| MLP_Silu_2h  |     6   |     7 |      4 |     5.7 |      7 |
| Feat_Attn_2h |     8   |     5 |      8 |     7.1 |      8 |
| Feat_Attn_3h |     9   |     9 |      9 |     9   |      9 |
