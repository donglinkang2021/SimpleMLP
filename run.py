import torch
import hydra
from omegaconf import OmegaConf, DictConfig
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from tqdm import tqdm

def is_regression(dataset_cfg: DictConfig) -> bool:
    return dataset_cfg["_target_"].split(".")[-1].startswith("regress_")

def get_result_dir(cfg: DictConfig) -> str:
    output_dir = cfg.env.out_dir
    dataset = cfg.dataset._target_.split('.')[-1]
    model = cfg.model._target_.split('.')[-1]
    optimizer = cfg.optimizer._target_.split('.')[-1]
    return f"{output_dir}/{dataset}_{model}_{optimizer}"

class SimpleDataset(Dataset):
    def __init__(self, dataset_cfg: DictConfig):
        super().__init__()
        self.x, self.y, self.label = hydra.utils.instantiate(dataset_cfg)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        self.X = torch.stack([self.x, self.y], dim=1)
        if is_regression(dataset_cfg):
            self.label = torch.from_numpy(self.label).float().unsqueeze(1)
        else:
            self.label = torch.from_numpy((self.label + 1) // 2).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]

@hydra.main(config_path="configs", config_name="config", version_base=None)
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # return
    import json
    from pathlib import Path
    res_dir = get_result_dir(cfg)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{res_dir}/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    is_regress = is_regression(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model, output_dim=1 if is_regress else 2)
    if cfg.train.init_weights:
        from utils import init_weights
        init_weights(model)
    dataset = SimpleDataset(cfg.dataset)
    trainset, valset, testset = torch.utils.data.random_split(dataset, cfg.train.splits)
    def get_loader(dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    trainloader, valloader, testloader = map(get_loader, [trainset, valset, testset])
    optimizer:torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters(), lr=cfg.train.learning_rate)
    criterion = torch.nn.MSELoss() if is_regress else torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def estimate():
        out = {}
        model.eval()
        for name, loader in [('train', trainloader), ('val', valloader), ('test', testloader)]:
            losses = []
            for x, y in loader:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out
    
    metrics = []
    n_batches = len(trainloader)
    pbar = tqdm(total=cfg.train.num_epochs * n_batches, dynamic_ncols=True, desc="Training")
    for epoch in range(cfg.train.num_epochs):
        for i, (x, y) in enumerate(trainloader):
            step = epoch * n_batches + i
            if step % cfg.train.eval_interval == 0 or step == cfg.train.num_epochs * n_batches - 1:
                metric = estimate()
                metric["step"] = step
                metrics.append(metric)
                pbar.set_postfix(metric)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    pbar.close()
    with open(f"{res_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    

if __name__ == "__main__":
    my_app()

"""
python run.py --multirun dataset=regress_plane,regress_gaussian,classify_two_gauss,classify_spiral,classify_circle,classify_xor model=mlp_relu_1h,mlp_relu_2h,mlp_tanh_1h,mlp_tanh_2h,mlp_silu_1h,mlp_silu_2h,feat_attn_1h,feat_attn_2h,feat_attn_3h optimizer=adam,sgd
"""