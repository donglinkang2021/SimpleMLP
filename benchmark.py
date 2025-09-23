import wandb
import swanlab
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
from pathlib import Path
import json

def get_project_exps_swanlab(proj_name:str, api:swanlab.OpenApi):
    """use api to get all the data"""
    data = []
    exps = api.list_experiments(project=proj_name).data
    print(f"  {len(exps)} experiments")
    for exp in tqdm(exps, desc="Fetching experiments summaries", dynamic_ncols=True):
        summary = api.get_summary(project=proj_name, exp_id=exp.cuid).data
        data.append({
            "id": exp.cuid, 
            "name": exp.name, 
            "state": exp.state,
            "profile": exp.profile, 
            "summary": summary
        })
    return data

def get_project_exps_wandb(path:str, api:wandb.Api):
    """use api to get all the data"""
    data = []
    runs = api.runs(path)
    print(f"  {len(runs)} experiments")
    for run in tqdm(runs, desc="Fetching runeriments summaries", dynamic_ncols=True):
        data.append({
            "id": run.id, 
            "name": run.name, 
            "state": run.state,
            "config": run.config, 
            "summary": run.summary
        })
    return data

def analyze_metrics(metrics: list):
    """
    Analyzes metrics and prints a separate academic-style table for each panel.
    Rows are models, and columns are datasets.
    """
    if not metrics:
        print("No metrics to analyze.")
        return

    df = pd.DataFrame(metrics)
    
    # Find all unique metric panels by looking for the 'loss' suffix
    metric_panels = {col for col in df.columns if col.endswith('loss')}

    for panel in sorted(list(metric_panels)):
        metric_col = panel
        
        # Pivot the DataFrame to get models as rows and datasets as columns
        pivot_df = df.pivot(index='model', columns='dataset', values=metric_col)
        
        # Find the best model (minimum value) for each dataset
        best_models = pivot_df.idxmin()

        # Format values to string and bold the best one
        for dataset in pivot_df.columns:
            best_model_for_dataset = best_models[dataset]
            pivot_df[dataset] = pivot_df[dataset].apply(
                lambda x: f"{x:.5f}" if pd.notna(x) else "N/A"
            )
            if pd.notna(best_model_for_dataset):
                pivot_df.loc[best_model_for_dataset, dataset] = f"**{pivot_df.loc[best_model_for_dataset, dataset]}**"

        print(f"\n--- Table for Metric: {panel} ---")
        print(tabulate(pivot_df, headers='keys', tablefmt='github'))

def rank_and_score_models(metrics: list):
    """
    Ranks models within each dataset, calculates an overall score, and prints a final ranking table.
    The score is a weighted average of the model's rank on train, validation, and test sets.
    Score = 0.2 * Avg_Train_Rank + 0.3 * Avg_Val_Rank + 0.5 * Avg_Test_Rank
    """
    if not metrics:
        print("No metrics to rank.")
        return

    df = pd.DataFrame(metrics)

    # Rename columns for easier access and clarity
    df.rename(columns={
        'eval/train_loss': 'train_loss',
        'eval/val_loss': 'val_loss',
        'eval/test_loss': 'test_loss'
    }, inplace=True)

    # Ensure all required loss columns are present
    required_cols = ['model', 'dataset', 'train_loss', 'val_loss', 'test_loss']
    if not all(col in df.columns for col in required_cols):
        print("Error: One or more required columns (train_loss, val_loss, test_loss) are missing.")
        return

    # Rank models within each dataset (lower loss is better)
    rank_frames = []
    for dataset in df["dataset"].unique():
        sub_df = df[df["dataset"] == dataset].copy()
        for split in ("train", "val", "test"):
            loss_col = f"{split}_loss"
            rank_col = f"{split}_rank"
            # Use 'min' method to handle ties correctly
            sub_df[rank_col] = sub_df[loss_col].rank(method="min", ascending=True)
        rank_frames.append(sub_df)
    
    if not rank_frames:
        print("No data to rank.")
        return
        
    rank_df = pd.concat(rank_frames, ignore_index=True)

    # Calculate average rank and final score for each model
    final_scores = []
    for model in rank_df["model"].unique():
        model_ranks = rank_df[rank_df["model"] == model]
        avg_train_rank = model_ranks["train_rank"].mean()
        avg_val_rank = model_ranks["val_rank"].mean()
        avg_test_rank = model_ranks["test_rank"].mean()
        
        # Weighted score, giving more importance to the test set
        score = (0.2 * avg_train_rank) + (0.3 * avg_val_rank) + (0.5 * avg_test_rank)
        
        final_scores.append({
            "Model": model,
            "Avg Train Rank": avg_train_rank,
            "Avg Val Rank": avg_val_rank,
            "Avg Test Rank": avg_test_rank,
            "Score": score,
        })

    # Create and sort the final DataFrame by score
    score_df = pd.DataFrame(final_scores).sort_values("Score").reset_index(drop=True)
    score_df["Final Rank"] = score_df.index + 1

    print("\n--- Overall Model Ranking ---")
    print(tabulate(score_df, headers="keys", tablefmt="github", floatfmt=".2f"))

def analyze_wandb():
    entity_name = "donglinkang2021-beijing-institute-of-technology"
    proj_name = "simplemlp"
    Path(f"results/{proj_name}").mkdir(exist_ok=True, parents=True)
    result_file = f"results/{proj_name}/metrics_wandb.json"
    
    if not Path(result_file).exists():
        data = get_project_exps_wandb(f"{entity_name}/{proj_name}", wandb.Api())
        metrics = [
            {
                "model": exp["config"]["model"]["_target_"].split(".")[-1],
                "dataset": exp["config"]["dataset"]["_target_"].split(".")[-1],
                **{k: v for k, v in exp["summary"].items() if k.endswith("loss")}
            }
            for exp in data if exp["state"] == "finished"
        ]
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        with open(result_file, "r") as f:
            metrics = json.load(f)
    
    analyze_metrics(metrics)
    rank_and_score_models(metrics)

def analyze_swanlab():
    proj_name = "simplemlp"
    Path(f"results/{proj_name}").mkdir(exist_ok=True, parents=True)
    result_file = f"results/{proj_name}/metrics_swanlab.json"
    
    if not Path(result_file).exists():
        data = get_project_exps_swanlab(proj_name, swanlab.OpenApi())
        metrics = [
            {
                "model": exp["profile"]["config"]["model"]["value"]["_target_"].split(".")[-1],
                "dataset": exp["profile"]["config"]["dataset"]["value"]["_target_"].split(".")[-1],
                **{pannel_name: summary["value"] for pannel_name, summary in exp["summary"].items()}
            }
            for exp in data if exp["state"] == "FINISHED"
        ]
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        with open(result_file, "r") as f:
            metrics = json.load(f)
    
    analyze_metrics(metrics)
    rank_and_score_models(metrics)

if __name__ == "__main__":
    analyze_swanlab()
    analyze_wandb()

# python benchmark.py