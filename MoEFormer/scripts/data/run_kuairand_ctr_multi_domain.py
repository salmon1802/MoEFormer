import sys
sys.path.append("..")
import os
import time
import json
import csv
import statistics
import logging
from datetime import datetime
from typing import Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import (
    MoEFormer
)

# -------------------------
# Logging / Stats Utilities
# -------------------------

def setup_logging(log_dir: str, run_name: str, rank_zero: bool = True) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if rank_zero:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = 0
    embed = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        lname = name.lower()
        if ("embed" in lname) or ("embedding" in lname) or ("emb_" in lname):
            embed += n
    net = total - embed
    return {"total": total, "embedding": embed, "network": net}


def get_device_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        def sync():
            torch.cuda.synchronize(device=torch.device(device))
        return sync
    else:
        def sync():
            return
        return sync


def benchmark_inference(model: torch.nn.Module,
                        dataloader,
                        device: str,
                        warmup: int = 3,
                        iters: int = 10,
                        logger: logging.Logger = None) -> Dict[str, Any]:
    model.eval()
    sync = get_device_sync(device)
    times = []
    n_samples = 0

    def run_epoch_like(max_iters):
        nonlocal n_samples
        it = iter(dataloader)
        i = 0
        while i < max_iters:
            try:
                batch_x, batch_y = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch_x, batch_y = next(it)

            # Move to device
            batch_x_on_dev = {}
            for k, v in batch_x.items():
                if isinstance(v, torch.Tensor):
                    batch_x_on_dev[k] = v.to(device)
                else:
                    try:
                        t = torch.as_tensor(v, device=device)
                        batch_x_on_dev[k] = t
                    except Exception:
                        batch_x_on_dev[k] = v
            batch_y_on_dev = batch_y.to(device) if isinstance(batch_y, torch.Tensor) else torch.as_tensor(batch_y, device=device)

            with torch.no_grad():
                sync()
                t0 = time.perf_counter()
                _ = model(batch_x_on_dev)
                sync()
                t1 = time.perf_counter()
            times.append(t1 - t0)

            bs = batch_y_on_dev.shape[0] if isinstance(batch_y_on_dev, torch.Tensor) else len(batch_y_on_dev)
            n_samples += int(bs)
            i += 1

    if warmup > 0:
        for _ in range(warmup):
            run_epoch_like(1)
        times.clear()

    run_epoch_like(iters)

    if len(times) == 0:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None, "samples_per_s": None, "iters": 0}

    avg = sum(times) / len(times)
    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=100)[94] if len(times) >= 20 else max(times)
    avg_ms = avg * 1000.0
    p50_ms = p50 * 1000.0
    p95_ms = p95 * 1000.0
    samples_per_s = n_samples / sum(times) if sum(times) > 0 else None

    result = {
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "samples_per_s": samples_per_s,
        "iters": len(times)
    }
    if logger:
        logger.info(f"Inference benchmark: iters={len(times)} avg={avg_ms:.3f} ms | p50={p50_ms:.3f} ms | p95={p95_ms:.3f} ms | samples/s={samples_per_s:.2f}")
    return result

# -------------------------
# Data Loading
# -------------------------

def convert_numeric(val):
    return int(val)

def get_kuairand_data_multidomain(data_path="./data/kuairand/"):
    """
    读取并预处理 Kuairand 数据。
    第一次运行会进行完整预处理并将结果缓存到 data_path/kuairand_preprocessed.pkl，
    之后运行如果该文件存在则直接读取缓存，跳过预处理。
    """
    os.makedirs(data_path, exist_ok=True)
    cache_path = os.path.join(data_path, "kuairand_preprocessed.pkl")

    # 如果有缓存，直接加载
    if os.path.exists(cache_path):
        cache = pd.read_pickle(cache_path)
        data = cache["data"]
        y = cache["y"]
        dense_features = cache["dense_features"]
        sparse_features = cache["sparse_features"]
        scenario_features = cache["scenario_features"]
        embed_dim = cache["embed_dim"]
        domain_num = cache["domain_num"]
    else:
        # ------------------ 正常预处理流程 ------------------
        raw_csv_path = os.path.join(data_path, "kuairand.csv")
        data = pd.read_csv(raw_csv_path)
        data = data[data["tab"].apply(lambda x: x in [1, 0, 4, 2, 6])]
        data.reset_index(drop=True, inplace=True)

        data.rename(columns={'tab': "domain_indicator"}, inplace=True)
        domain_num = data.domain_indicator.nunique()
        embed_dim = 16

        col_names = data.columns.to_list()

        dense_features = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]

        useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
        scenario_features = ["domain_indicator"]

        sparse_features = [col for col in col_names if col not in dense_features and
                           col not in useless_features and col not in ['is_click', 'domain_indicator']]

        for feature in dense_features:
            data[feature] = data[feature].apply(lambda x: convert_numeric(x))
        if dense_features:
            sca = MinMaxScaler()
            data[dense_features] = sca.fit_transform(data[dense_features])

        for feature in useless_features:
            if feature in data.columns:
                del data[feature]

        for feature in scenario_features:
            lbe = LabelEncoder()
            data[feature] = lbe.fit_transform(data[feature])

        for feature in tqdm(sparse_features, desc="Encoding sparse features"):
            lbe = LabelEncoder()
            data[feature] = lbe.fit_transform(data[feature])

        y = data["is_click"]
        del data["is_click"]

        # 保存到缓存文件
        cache = {
            "data": data,
            "y": y,
            "dense_features": dense_features,
            "sparse_features": sparse_features,
            "scenario_features": scenario_features,
            "embed_dim": embed_dim,
            "domain_num": domain_num,
        }
        pd.to_pickle(cache, cache_path)
        print(f"[Data] Preprocessed data cached at: {cache_path}")

    # 下面部分无论是从缓存还是新处理都统一执行（构造 Feature 对象等）
    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=embed_dim)
                   for feature_name in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=int(data[col].max()) + 1, embed_dim=embed_dim)
                     for col in scenario_features]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

# -------------------------
# Main
# -------------------------

def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed,
         log_dir="./logs", infer_warmup=3, infer_bench_iters=10):
    # Logger
    run_name = f"{model_name}_kuairand_{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logger = setup_logging(log_dir, run_name)

    torch.manual_seed(seed)
    dataset_name = "Kuairand"
    logger.info(f"Args: dataset_path={dataset_path} model_name={model_name} epoch={epoch} lr={learning_rate} "
                f"batch_size={batch_size} weight_decay={weight_decay} device={device} save_dir={save_dir} seed={seed}")

    # Data（内部会自动判断是否使用缓存）
    dense_feas, sparse_feas, scenario_feas, x, y, domain_num = get_kuairand_data_multidomain(dataset_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1],
                                                                               batch_size=batch_size)

    model = MoEFormer(sparse_feas, domain_num, hidden_dim=1, num_layers=2, num_semantic_tokens=64, num_domain_tokens=4, expand_ratio=1)

    # Device
    dev = torch.device(device)
    model = model.to(dev)

    # Param stats
    param_stats = count_parameters(model)
    logger.info(f"Parameters: total={param_stats['total']:,} | embedding={param_stats['embedding']:,} | network={param_stats['network']:,}")

    # Trainer
    ctr_trainer = CTRTrainer(
        model,
        dataset_name,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=1,
        device=device,
        model_path=save_dir,
        scheduler_params={"step_size": 4, "gamma": 0.75}
    )

    # Train
    logger.info("Start training...")
    ctr_trainer.fit(train_dataloader, val_dataloader)
    logger.info("Training finished")

    # Eval
    logger.info("Start evaluation...")
    domain_logloss, domain_auc, logloss, auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader, domain_num)
    logger.info(f"Test auc: {auc:.6f} | Test logloss: {logloss:.6f}")
    for d in range(domain_num):
        logger.info(f"Domain {d} auc: {domain_auc[d]:.6f} | Domain {d} logloss: {domain_logloss[d]:.6f}")

    # Inference benchmark
    logger.info("Benchmarking inference latency on test set batches...")
    infer_stats = benchmark_inference(ctr_trainer.model, test_dataloader, device=device,
                                      warmup=infer_warmup, iters=infer_bench_iters, logger=logger)

    # Ensure save dir
    os.makedirs(save_dir, exist_ok=True)

    # CSV summary (fixed 5 domains for current Kuairand filter)
    csv_path = os.path.join(save_dir, f"{model_name}_Kuairand_{str(seed)}.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'seed', 'auc', 'log',
            'auc0', 'log0', 'auc1', 'log1', 'auc2', 'log2', 'auc3', 'log3', 'auc4', 'log4',
            'total_params', 'embed_params', 'network_params',
            'infer_avg_ms', 'infer_p50_ms', 'infer_p95_ms', 'samples_per_s'
        ])
        def get_metric(lst, i):
            return lst[i] if (isinstance(lst, (list, tuple)) and len(lst) > i) else None
        writer.writerow([
            model_name, str(seed), auc, logloss,
            get_metric(domain_auc, 0), get_metric(domain_logloss, 0),
            get_metric(domain_auc, 1), get_metric(domain_logloss, 1),
            get_metric(domain_auc, 2), get_metric(domain_logloss, 2),
            get_metric(domain_auc, 3), get_metric(domain_logloss, 3),
            get_metric(domain_auc, 4), get_metric(domain_logloss, 4),
            param_stats["total"], param_stats["embedding"], param_stats["network"],
            infer_stats.get("avg_ms"), infer_stats.get("p50_ms"), infer_stats.get("p95_ms"), infer_stats.get("samples_per_s")
        ])
    logger.info(f"Saved CSV to {csv_path}")

    # JSON summary (dynamic)
    json_path = os.path.join(save_dir, f"{model_name}_Kuairand_{str(seed)}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "model": model_name,
            "seed": seed,
            "metrics": {
                "auc": auc, "logloss": logloss,
                "domain_auc": domain_auc, "domain_logloss": domain_logloss
            },
            "params": param_stats,
            "inference": infer_stats
        }, jf, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON to {json_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/kuairand")
    parser.add_argument('--model_name', default='MoEFormer')
    parser.add_argument('--epoch', type=int, default=1)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--device', default='cuda:5')  # cuda:0
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--seed', type=int, default=20252026)

    # Aligned logging/benchmark args
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--infer_warmup', type=int, default=3)
    parser.add_argument('--infer_bench_iters', type=int, default=10)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay,
         args.device, args.save_dir, args.seed, args.log_dir, args.infer_warmup, args.infer_bench_iters)