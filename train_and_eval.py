import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader


# ========== Configuration ==========

@dataclass
class RunConfig:
    base_data_path: str
    base_emb_path: str
    base_result_path: str
    save_dir: str
    best_hyperparams_file: str

    seed: int = 42
    lr: float = 1e-4
    batch_size: int = 32

    hard_negative_topk: int = 1000
    num_negatives: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    overwrite: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a predictor for one or multiple model/dataset pairs."
    )

    parser.add_argument("--models", nargs="+", required=True, help="Model names, e.g. qwen-4b")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names, e.g. scifact")

    parser.add_argument("--base-data-path", type=str, default="/home/wuzy24/models/beir-dataset")
    parser.add_argument("--base-emb-path", type=str, default="../code/data/embeddings")
    parser.add_argument("--base-result-path", type=str, default="results/predictor_runs")
    parser.add_argument("--save-dir", type=str, default="artifacts/predictors")
    parser.add_argument("--best-hyperparams-file", type=str, default="best_paths_hyperparams.json")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negative-topk", type=int, default=1000)
    parser.add_argument("--num-negatives", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# ========== Utility ==========

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✅ Random seed set to {seed}")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


def build_id_index(ids: List[str]) -> Dict[str, int]:
    return {id_: i for i, id_ in enumerate(ids)}


def ndcg_at_k(rel_list: List[int], k: int = 10) -> float:
    def dcg(scores):
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))

    ideal = sorted(rel_list, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(rel_list) / idcg


def save_json(path: str, obj: Dict):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dataset_name_for_embedding(dataset_name: str) -> str:
    return dataset_name.replace("_LLM", "").replace("_hard", "")


def build_eval_schedule_by_percent(total_dim: int, step_percent: int = 10) -> List[Tuple[int, int]]:
    """
    Return a list of (percent, k_dims), e.g.
    [(10, 77), (20, 154), ..., (100, 768)]
    """
    schedule = []
    used_dims = set()

    for percent in range(step_percent, 101, step_percent):
        k_dims = max(1, math.ceil(total_dim * percent / 100.0))
        k_dims = min(k_dims, total_dim)

        if k_dims not in used_dims:
            schedule.append((percent, k_dims))
            used_dims.add(k_dims)
        else:
            # In rare low-dim cases, different percent may map to same k_dims.
            # Still keep the percent entry for consistency.
            schedule.append((percent, k_dims))

    if schedule[-1][1] != total_dim:
        schedule[-1] = (100, total_dim)

    return schedule


# ========== Model Definition ==========

class DimensionPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)


# ========== Dataset ==========

class PredictorDataset(Dataset):
    def __init__(self, query_embeddings: torch.Tensor, target_probs: torch.Tensor):
        self.q_emb = query_embeddings
        self.targets = target_probs

    def __len__(self):
        return len(self.q_emb)

    def __getitem__(self, idx):
        return self.q_emb[idx], self.targets[idx]


# ========== Training Label Generation ==========

@torch.no_grad()
def generate_soft_oracle_labels_multilevel(
    query_ids,
    corpus_ids,
    q_emb,
    d_emb,
    qrels,
    temperature: float = 0.1,
    K: int = 1000,
    M: int = 64,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    Q, D = q_emb.shape
    N = d_emb.shape[0]

    q_mat = q_emb.to(device).float()
    d_mat = d_emb.to(device).float()

    q_mat = l2_normalize_rows(q_mat)
    d_mat = l2_normalize_rows(d_mat)

    id2idx = build_id_index(corpus_ids)

    valid_q_indices = []
    target_probs_list = []

    qidx2pos_info = {}
    for qi, qid in enumerate(query_ids):
        rel_docs = []
        rel_scores = []
        for doc_id, rel in qrels.get(qid, {}).items():
            if rel > 0 and doc_id in id2idx:
                idx = id2idx[doc_id]
                if idx < N:
                    rel_docs.append(idx)
                    rel_scores.append(float(rel))

        if len(rel_docs) > 0:
            qidx2pos_info[qi] = {
                "indices": torch.tensor(rel_docs, device=device, dtype=torch.long),
                "scores": torch.tensor(rel_scores, device=device, dtype=torch.float),
            }

    for qi in range(Q):
        if qi not in qidx2pos_info:
            continue

        info = qidx2pos_info[qi]
        pos_idx = info["indices"]
        pos_scores = info["scores"]

        # 1. Gain-based weighted positive mean
        weights = torch.pow(2.0, pos_scores) - 1.0
        weights = weights / (weights.sum() + 1e-9)
        weights = weights.unsqueeze(1)

        pos_vecs = d_mat[pos_idx]
        weighted_pos_mean = (pos_vecs * weights).sum(dim=0)

        # 2. Hard negative mining
        sim_scores = d_mat @ q_mat[qi]
        k_hard = min(K, N)
        top_candidates = torch.topk(sim_scores, k=k_hard).indices
        mask_is_pos = torch.isin(top_candidates, pos_idx)
        hard_neg_pool = top_candidates[~mask_is_pos]

        num_neg = M
        if len(hard_neg_pool) >= num_neg:
            sampled_idx = torch.randint(0, len(hard_neg_pool), (num_neg,), device=device)
            neg_idx = hard_neg_pool[sampled_idx]
        else:
            if len(hard_neg_pool) > 0:
                remaining = num_neg - len(hard_neg_pool)
                rand_neg = torch.randint(0, N, (remaining,), device=device)
                neg_idx = torch.cat([hard_neg_pool, rand_neg])
            else:
                neg_idx = torch.randint(0, N, (num_neg,), device=device)

        neg_vecs = d_mat[neg_idx]
        neg_mean = neg_vecs.mean(dim=0)

        # 3. Dimension-level target distribution
        delta = weighted_pos_mean - neg_mean
        raw_scores = delta * q_mat[qi]
        probs = F.softmax(raw_scores / temperature, dim=0)

        valid_q_indices.append(qi)
        target_probs_list.append(probs.cpu())

    if not valid_q_indices:
        raise ValueError("No valid queries found for training.")

    train_q_emb = q_emb[valid_q_indices].clone()
    train_targets = torch.stack(target_probs_list)

    return train_q_emb, train_targets


# ========== Training ==========

def train_predictor(
    model,
    train_loader,
    epochs: int = 5,
    device: str = "cuda",
    lr: float = 1e-4,
):
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch_idx in range(epochs):
        model.train()
        total_train_loss = 0.0

        for q_batch, target_batch in train_loader:
            q_batch = q_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            log_probs = model(q_batch)
            loss = criterion(log_probs, target_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        if epoch_idx % 10 == 0 or epoch_idx == epochs - 1:
            print(f"    Epoch {epoch_idx + 1}/{epochs} | train_loss={avg_train_loss:.6f}")

    return model, avg_train_loss


# ========== Evaluation ==========

@torch.no_grad()
def evaluate_predictor_strict(
    predictor,
    query_ids,
    corpus_ids,
    q_emb,
    d_emb,
    qrels,
    device: str = "cuda",
):
    predictor.eval()
    predictor.to(device)

    Q, D = q_emb.shape
    N = d_emb.shape[0]

    # Evaluate at every 10% dimensions: 10%, 20%, ..., 100%
    eval_schedule = build_eval_schedule_by_percent(total_dim=D, step_percent=10)

    q_mat = q_emb.to(device).float()
    d_mat = d_emb.to(device).float()

    q_mat = l2_normalize_rows(q_mat)
    d_mat = l2_normalize_rows(d_mat)

    # Batch predict
    all_log_probs = []
    pred_batch_size = 1024
    for i in range(0, Q, pred_batch_size):
        batch_q = q_mat[i:i + pred_batch_size]
        log_probs = predictor(batch_q)
        all_log_probs.append(log_probs)
    all_log_probs = torch.cat(all_log_probs, dim=0)

    results_predictor_by_percent = {str(percent): [] for percent, _ in eval_schedule}
    results_baseline_by_percent = {str(percent): [] for percent, _ in eval_schedule}

    results_predictor_by_dim = {str(k_dims): [] for _, k_dims in eval_schedule}
    results_baseline_by_dim = {str(k_dims): [] for _, k_dims in eval_schedule}

    id2idx = build_id_index(corpus_ids)
    valid_q_indices = []
    for qi, qid in enumerate(query_ids):
        rel_docs = [doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0]
        pos_idx = [id2idx[d] for d in rel_docs if d in id2idx]
        if len(pos_idx) > 0:
            valid_q_indices.append(qi)

    for qi in tqdm(valid_q_indices, desc="  Eval", leave=False):
        qid = query_ids[qi]
        q_vec = q_mat[qi]
        dim_scores = all_log_probs[qi]

        def calc_strict_ndcg(scores_tensor):
            topN = torch.topk(scores_tensor, k=min(N, 2000), largest=True)
            ranked_idx = topN.indices.tolist()

            rel_list = []
            qrel_dict = qrels.get(qid, {})
            for di in ranked_idx[:1000]:
                doc_id = corpus_ids[di]
                rel_list.append(qrel_dict.get(doc_id, 0))

            return ndcg_at_k(rel_list, k=10)

        for percent, k_dims in eval_schedule:
            # Predictor-selected dimensions
            topk_predictor = torch.topk(dim_scores, k=k_dims).indices
            q_sub_p = q_vec[topk_predictor]
            d_sub_p = d_mat[:, topk_predictor]
            scores_predictor = d_sub_p @ q_sub_p

            # Baseline: first k dimensions
            topk_base = torch.arange(k_dims, device=device)
            q_sub_b = q_vec[topk_base]
            d_sub_b = d_mat[:, topk_base]
            scores_baseline = d_sub_b @ q_sub_b

            predictor_score = calc_strict_ndcg(scores_predictor)
            baseline_score = calc_strict_ndcg(scores_baseline)

            results_predictor_by_percent[str(percent)].append(predictor_score)
            results_baseline_by_percent[str(percent)].append(baseline_score)

            results_predictor_by_dim[str(k_dims)].append(predictor_score)
            results_baseline_by_dim[str(k_dims)].append(baseline_score)

    avg_predictor_by_percent = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0
        for k, v in results_predictor_by_percent.items()
    }
    avg_baseline_by_percent = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0
        for k, v in results_baseline_by_percent.items()
    }

    avg_predictor_by_dim = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0
        for k, v in results_predictor_by_dim.items()
    }
    avg_baseline_by_dim = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0
        for k, v in results_baseline_by_dim.items()
    }

    return {
        "schedule": [{"percent": percent, "dim": k_dims} for percent, k_dims in eval_schedule],
        "predictor_by_percent": avg_predictor_by_percent,
        "baseline_by_percent": avg_baseline_by_percent,
        "predictor_by_dim": avg_predictor_by_dim,
        "baseline_by_dim": avg_baseline_by_dim,
    }


# ========== Data Loading ==========

def load_data_and_embeddings(model_name: str, dataset_name: str, config: RunConfig):
    clean_dataset_name = normalize_dataset_name_for_embedding(dataset_name)

    corpus_train, queries_train, qrels_train = GenericDataLoader(
        data_folder=f"{config.base_data_path}/{dataset_name}"
    ).load(split="train")

    corpus_test, queries_test, qrels_test = GenericDataLoader(
        data_folder=f"{config.base_data_path}/{dataset_name}"
    ).load(split="test")

    q_emb_train = torch.load(
        f"{config.base_emb_path}/{model_name}/{dataset_name}/train/queries_embeddings.pt"
    )

    d_emb_test = torch.load(
        f"{config.base_emb_path}/{model_name}/{clean_dataset_name}/test/corpus_embeddings.pt"
    )

    try:
        d_emb_train = torch.load(
            f"{config.base_emb_path}/{model_name}/{clean_dataset_name}/train/corpus_embeddings.pt"
        )
    except FileNotFoundError:
        d_emb_train = d_emb_test

    q_emb_test = torch.load(
        f"{config.base_emb_path}/{model_name}/{clean_dataset_name}/test/queries_embeddings.pt"
    )

    return (
        corpus_train, queries_train, qrels_train,
        corpus_test, queries_test, qrels_test,
        q_emb_train, q_emb_test, d_emb_train, d_emb_test,
    )


# ========== Single Run ==========

def run_single_experiment(model_name: str, dataset_name: str, hyps: Dict, config: RunConfig):
    print(f"\n{'=' * 80}")
    print(f">>> Processing: {model_name} / {dataset_name}")
    print(f"{'=' * 80}")

    if model_name not in hyps or dataset_name not in hyps[model_name]:
        print(f"❌ Missing hyperparameters for {model_name} / {dataset_name} in {config.best_hyperparams_file}")
        return

    epoch = hyps[model_name][dataset_name]["epoch"]
    temp = hyps[model_name][dataset_name]["temp"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError(f"Invalid epoch for {model_name} / {dataset_name}: {epoch!r}")
    if not isinstance(temp, (int, float)) or isinstance(temp, bool) or temp <= 0:
        raise ValueError(f"Invalid temp for {model_name} / {dataset_name}: {temp!r}")

    predictor_path = os.path.join(config.save_dir, model_name, f"{dataset_name}.pt")
    result_file = os.path.join(config.base_result_path, model_name, f"{dataset_name}.json")

    if (not config.overwrite) and os.path.exists(predictor_path) and os.path.exists(result_file):
        print("⏭️  Predictor and result already exist, skipping.")
        print(f"   Predictor: {predictor_path}")
        print(f"   Result:    {result_file}")
        return

    try:
        (
            corpus_train, queries_train, qrels_train,
            corpus_test, queries_test, qrels_test,
            q_emb_train, q_emb_test, d_emb_train, d_emb_test,
        ) = load_data_and_embeddings(model_name, dataset_name, config)
    except FileNotFoundError as e:
        print(f"❌ Missing file for {model_name} / {dataset_name}")
        print(f"   {e}")
        return

    corpus_ids = sorted(
        corpus_test,
        key=lambda k: len(corpus_test[k].get("title", "") + corpus_test[k].get("text", "")),
        reverse=True,
    )

    train_q_ids = list(queries_train.keys())
    test_q_ids = list(queries_test.keys())

    print(f"  > Generating labels (temperature={temp})...")
    X_train, Y_train = generate_soft_oracle_labels_multilevel(
        query_ids=train_q_ids,
        corpus_ids=corpus_ids,
        q_emb=q_emb_train,
        d_emb=d_emb_train,
        qrels=qrels_train,
        temperature=temp,
        K=config.hard_negative_topk,
        M=config.num_negatives,
        device=config.device,
    )

    dataset = PredictorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    print(f"  > Training predictor (epochs={epoch})...")
    predictor = DimensionPredictor(input_dim=q_emb_train.shape[1]).to(config.device)
    predictor, final_train_loss = train_predictor(
        model=predictor,
        train_loader=train_loader,
        epochs=epoch,
        device=config.device,
        lr=config.lr,
    )

    ensure_parent_dir(predictor_path)
    torch.save(
        {
            "model_state_dict": predictor.state_dict(),
            "input_dim": q_emb_train.shape[1],
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epoch": epoch,
            "temperature": temp,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "hard_negative_topk": config.hard_negative_topk,
            "num_negatives": config.num_negatives,
            "seed": config.seed,
            "final_train_loss": final_train_loss,
        },
        predictor_path,
    )
    print(f"  ✅ Predictor saved to: {predictor_path}")

    print("  > Evaluating...")
    eval_results = evaluate_predictor_strict(
        predictor=predictor,
        query_ids=test_q_ids,
        corpus_ids=corpus_ids,
        q_emb=q_emb_test,
        d_emb=d_emb_test,
        qrels=qrels_test,
        device=config.device,
    )

    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "hyperparams": {
            "epoch": epoch,
            "temperature": temp,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "hard_negative_topk": config.hard_negative_topk,
            "num_negatives": config.num_negatives,
            "seed": config.seed,
            "device": config.device,
        },
        "artifacts": {
            "predictor_path": predictor_path,
        },
        "training": {
            "num_training_queries": len(train_q_ids),
            "num_labeled_queries": len(dataset),
            "final_train_loss": final_train_loss,
        },
        "evaluation": {
            "schedule": eval_results["schedule"],
            "step_percent": 10,
        },
        "metrics": {
            "by_percent": {
                "predictor": eval_results["predictor_by_percent"],
                # "baseline": eval_results["baseline_by_percent"],
                # "gain": gain_by_percent,
            },
            "by_dim": {
                "predictor": eval_results["predictor_by_dim"],
                # "baseline": eval_results["baseline_by_dim"],
                # "gain": gain_by_dim,
            },
        },
    }

    save_json(result_file, result)
    print(f"  ✅ Result saved to: {result_file}")
    print(f"✅ Completed: {model_name} / {dataset_name}")


# ========== Main ==========

def main():
    args = parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    config = RunConfig(
        base_data_path=args.base_data_path,
        base_emb_path=args.base_emb_path,
        base_result_path=args.base_result_path,
        save_dir=args.save_dir,
        best_hyperparams_file=args.best_hyperparams_file,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        hard_negative_topk=args.hard_negative_topk,
        num_negatives=args.num_negatives,
        device=device,
        overwrite=args.overwrite,
    )

    set_random_seed(config.seed)

    hyps = load_json(config.best_hyperparams_file)

    for model_name in args.models:
        for dataset_name in args.datasets:
            run_single_experiment(model_name, dataset_name, hyps, config)


if __name__ == "__main__":
    main()
