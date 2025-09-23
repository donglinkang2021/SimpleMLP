<div align="center">

# Simple MLP

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
\text{Score} = 0.2 \times \text{train\_rank} + 0.3 \times \text{val\_rank} + 0.5 \times \text{test\_rank}
$$

The details can be found in the scripts `benchmark.py`.

--- Table for Metric: eval/test_loss ---
| model        | classify_circle   | classify_spiral   | classify_two_gauss   | classify_xor   | regress_gaussian   | regress_plane   |
|--------------|-------------------|-------------------|----------------------|----------------|--------------------|-----------------|
| Feat_Attn_1h | 0.13828           | 0.64182           | 0.01883              | 0.09869        | 0.24072            | 0.00603         |
| Feat_Attn_2h | 0.08670           | **0.10794**       | 0.01374              | 0.20605        | 0.19263            | 0.00608         |
| Feat_Attn_3h | 0.11397           | 0.58535           | 0.01143              | 0.09584        | 0.24249            | **0.00546**     |
| MLP_Relu_1h  | 0.14399           | 0.64204           | **0.01026**          | 0.17061        | 0.24586            | 0.00765         |
| MLP_Relu_2h  | 0.08216           | 0.50265           | 0.01724              | 0.13503        | 0.27684            | 0.00695         |
| MLP_Silu_1h  | **0.07326**       | 0.64828           | 0.06772              | 0.09633        | 0.33006            | 0.00622         |
| MLP_Silu_2h  | 0.09902           | 0.63212           | 0.02253              | **0.07824**    | **0.17893**        | 0.00662         |
| MLP_Tanh_1h  | 0.22521           | 0.66980           | 0.03845              | 0.21976        | 0.23246            | 0.00693         |
| MLP_Tanh_2h  | 0.11693           | 0.55830           | 0.04512              | 0.24656        | 0.18215            | 0.00780         |

--- Table for Metric: eval/train_loss ---
| model        | classify_circle   | classify_spiral   | classify_two_gauss   | classify_xor   | regress_gaussian   | regress_plane   |
|--------------|-------------------|-------------------|----------------------|----------------|--------------------|-----------------|
| Feat_Attn_1h | 0.14235           | 0.56710           | 0.03453              | 0.11425        | 0.24908            | **0.00571**     |
| Feat_Attn_2h | 0.09466           | **0.09915**       | **0.02704**          | **0.08981**    | **0.18101**        | 0.00585         |
| Feat_Attn_3h | 0.11366           | 0.58333           | 0.03290              | 0.12126        | 0.21809            | 0.00667         |
| MLP_Relu_1h  | **0.08911**       | 0.63954           | 0.02938              | 0.13461        | 0.30452            | 0.00653         |
| MLP_Relu_2h  | 0.09699           | 0.49163           | 0.02729              | 0.09199        | 0.23628            | 0.00579         |
| MLP_Silu_1h  | 0.12086           | 0.64183           | 0.03984              | 0.10862        | 0.29370            | 0.00673         |
| MLP_Silu_2h  | 0.10570           | 0.62291           | 0.04550              | 0.12256        | 0.20667            | 0.00646         |
| MLP_Tanh_1h  | 0.22829           | 0.63983           | 0.03460              | 0.19291        | 0.23423            | 0.00675         |
| MLP_Tanh_2h  | 0.10807           | 0.61558           | 0.03499              | 0.12716        | 0.18256            | 0.00671         |

--- Table for Metric: eval/val_loss ---
| model        | classify_circle   | classify_spiral   | classify_two_gauss   | classify_xor   | regress_gaussian   | regress_plane   |
|--------------|-------------------|-------------------|----------------------|----------------|--------------------|-----------------|
| Feat_Attn_1h | 0.17352           | 0.56856           | 0.03180              | 0.15349        | 0.32270            | 0.00625         |
| Feat_Attn_2h | 0.14353           | **0.15648**       | 0.04208              | 0.09412        | 0.22759            | 0.00829         |
| Feat_Attn_3h | 0.10976           | 0.61685           | 0.01582              | 0.20033        | **0.19606**        | 0.00549         |
| MLP_Relu_1h  | 0.14376           | 0.66614           | 0.03148              | 0.32264        | 0.32193            | 0.00506         |
| MLP_Relu_2h  | 0.09164           | 0.46357           | 0.01722              | 0.13900        | 0.30670            | 0.00853         |
| MLP_Silu_1h  | 0.09286           | 0.69471           | 0.03010              | **0.08681**    | 0.34877            | 0.00715         |
| MLP_Silu_2h  | **0.07943**       | 0.60660           | 0.01709              | 0.13185        | 0.21620            | **0.00388**     |
| MLP_Tanh_1h  | 0.19793           | 0.68494           | **0.00873**          | 0.15240        | 0.27943            | 0.00493         |
| MLP_Tanh_2h  | 0.12129           | 0.68197           | 0.01591              | 0.11396        | 0.24743            | 0.00673         |

--- Table for Metric: train/loss ---
| model        | classify_circle   | classify_spiral   | classify_two_gauss   | classify_xor   | regress_gaussian   | regress_plane   |
|--------------|-------------------|-------------------|----------------------|----------------|--------------------|-----------------|
| Feat_Attn_1h | 0.11984           | 0.55508           | 0.00525              | 0.08681        | 0.18685            | 0.00408         |
| Feat_Attn_2h | 0.05040           | **0.07304**       | 0.00194              | 0.02762        | 0.13492            | **0.00341**     |
| Feat_Attn_3h | **0.03488**       | 0.51829           | **0.00042**          | 0.04753        | 0.14847            | 0.00496         |
| MLP_Relu_1h  | 0.05494           | 0.57261           | 0.00400              | 0.06420        | 0.25354            | 0.00545         |
| MLP_Relu_2h  | 0.03860           | 0.47677           | 0.00330              | **0.02279**    | 0.15606            | 0.00444         |
| MLP_Silu_1h  | 0.06138           | 0.59650           | 0.01206              | 0.06472        | 0.25416            | 0.00547         |
| MLP_Silu_2h  | 0.03956           | 0.54620           | 0.00586              | 0.05196        | 0.16505            | 0.00490         |
| MLP_Tanh_1h  | 0.18565           | 0.59127           | 0.00923              | 0.17082        | 0.18319            | 0.00572         |
| MLP_Tanh_2h  | 0.06754           | 0.56503           | 0.00383              | 0.04506        | **0.11740**        | 0.00459         |

--- Overall Model Ranking ---
|    | Model        |   Avg Train Rank |   Avg Val Rank |   Avg Test Rank |   Score |   Final Rank |
|----|--------------|------------------|----------------|-----------------|---------|--------------|
|  0 | Feat_Attn_2h |             1.50 |           4.83 |            3.33 |    3.42 |            1 |
|  1 | MLP_Silu_2h  |             5.33 |           2.67 |            3.67 |    3.70 |            2 |
|  2 | Feat_Attn_3h |             4.83 |           4.00 |            3.33 |    3.83 |            3 |
|  3 | MLP_Relu_2h  |             2.83 |           4.83 |            4.67 |    4.35 |            4 |
|  4 | Feat_Attn_1h |             4.67 |           6.50 |            4.83 |    5.30 |            5 |
|  5 | MLP_Tanh_2h  |             5.50 |           4.67 |            6.17 |    5.58 |            6 |
|  6 | MLP_Silu_1h  |             7.17 |           5.83 |            5.67 |    6.02 |            7 |
|  7 | MLP_Relu_1h  |             5.50 |           6.50 |            6.17 |    6.13 |            8 |
|  8 | MLP_Tanh_1h  |             7.67 |           5.17 |            7.17 |    6.67 |            9 |
