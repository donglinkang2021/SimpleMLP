uv run main.py --multirun \
    'model=glob(mlp_*)' \
    'dataset=glob(classify_*)' \
    optimizer=adamw \
    logger=wandb

uv run main.py --multirun \
    'model=glob(mlp_*)' \
    'dataset=glob(regress_*)' \
    optimizer=adamw \
    logger=wandb

uv run main.py --multirun \
    'model=glob(feat_attn_*)' \
    'dataset=glob(classify_*)' \
    optimizer=adamw \
    logger=wandb

uv run main.py --multirun \
    'model=glob(feat_attn_*)' \
    'dataset=glob(regress_*)' \
    optimizer=adamw \
    logger=wandb
