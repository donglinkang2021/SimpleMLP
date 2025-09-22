uv run main.py --multirun \
    dataset=regress_plane,regress_gaussian,classify_two_gauss,classify_spiral,classify_circle,classify_xor \
    model=mlp_relu_1h,mlp_relu_2h,mlp_tanh_1h,mlp_tanh_2h,mlp_silu_1h,mlp_silu_2h,feat_attn_1h,feat_attn_2h,feat_attn_3h \
    optimizer=adamw
