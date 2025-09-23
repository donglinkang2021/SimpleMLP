import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra_utils import register_omegaconf_resolvers
register_omegaconf_resolvers()

from logger import Logger

def is_regression(dataset_cfg: DictConfig) -> bool:
    return dataset_cfg["_target_"].split(".")[-1].startswith("regress_")

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
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return
    logger = Logger(cfg)

    is_regress = is_regression(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model, output_dim=1 if is_regress else 2)
    if cfg.train.init_weights:
        from utils import init_weights
        init_weights(model)
    dataset = SimpleDataset(cfg.dataset)
    trainset, valset, testset = torch.utils.data.random_split(dataset, cfg.train.splits)
    def get_loader(dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    trainloader_for_eval, valloader, testloader = map(get_loader, [trainset, valset, testset])

    def get_train_batch():
        indices = torch.randint(len(trainset), (cfg.train.batch_size,))
        x_list, y_list = zip(*[trainset[i] for i in indices])
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x, y

    optimizer:torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters(), lr=cfg.train.learning_rate)
    criterion = torch.nn.MSELoss() if is_regress else torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def estimate():
        out = {}
        model.eval()
        for name, loader in [
            ('train_loss', trainloader_for_eval), 
            ('val_loss', valloader), 
            ('test_loss', testloader)
        ]:
            losses = []
            for x, y in loader:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out
    
    for step in trange(cfg.train.max_steps, dynamic_ncols=True, desc="Training"):
        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
            metric = estimate()
            log_metric = {f"eval/{k}": v for k, v in metric.items()}
            logger.log_metrics(log_metric, step=step)
            tqdm.write(f"[STEP {step}]" + "".join([f" {k}: {v:.4f}" for k, v in metric.items()]))
        x, y = get_train_batch()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % cfg.train.eval_interval == 0:
            logger.log_metrics({"train/loss": loss.item()}, step=step)

    logger.close()

if __name__ == "__main__":
    my_app()
