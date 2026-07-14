"""Social Network Analysis with Graph Neural Networks.

Stages (run all by default, or pick with --stages):
  baselines   Logistic Regression on raw features; DeepWalk + LR
  models      GraphSAGE vs GCN vs GAT vs GIN (multi-seed)
  aggregators SAGE mean/max/sum aggregation vs GAT attention
  features    SNA structural-feature ablation (raw vs SNA vs raw+SNA)
  depth       1-4 message-passing layers (oversmoothing)
  labelrate   accuracy vs fraction of labeled training nodes
  sampling    fixed fan-out vs degree-adaptive S_i = c + [N_i * w]
  linkpred    link prediction (ROC-AUC / Average Precision)
  community   Louvain vs KMeans on raw features vs KMeans on embeddings

Usage:
  python main.py                       # everything, on Cora
  python main.py --stages models,community
  python main.py --dataset CiteSeer --quick
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import config
from config import (DEVICE, GNN_DROPOUT, GNN_EPOCHS, GNN_HIDDEN, GNN_LR,
                    GNN_PATIENCE, GNN_WEIGHT_DECAY, LINKPRED_EPOCHS,
                    LINKPRED_HIDDEN, LINKPRED_LR, LINKPRED_OUT, LR_MAX_ITER,
                    RESULTS_DIR, SEEDS)
from experiments.community_detection import community_detection_experiment
from experiments.feature_ablation import feature_ablation_experiment
from experiments.layer_experiments import layer_experiment
from experiments.model_comparison import (aggregator_experiment,
                                          model_comparison_experiment)
from experiments.sampling_experiments import sampling_rate_experiment
from models.deepwalk_model import deepwalk_embedding
from models.gnn_zoo import GNN
from models.link_predictor import run_link_prediction
from models.logistic_regression import train_eval_logistic
from utils.data_loader import load_planetoid, set_seeds
from utils.metrics import full_metrics
from utils.train_utils import train_node_classifier
from utils.visualization import (pca_and_plot, plot_confusion_matrix,
                                 plot_training_curves, tsne_and_plot)

ALL_STAGES = ["baselines", "models", "aggregators", "features", "depth",
              "labelrate", "sampling", "linkpred", "community"]


def _clean(obj):
    """Make results JSON-serializable (drop models/tensors, cast numpy types)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if k not in ("model", "runs", "logits", "history", "confusion")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def stage_baselines(dataset, data, args, results):
    print("\n=== Baselines: Logistic Regression + DeepWalk ===")
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    acc_lr, report_lr = train_eval_logistic(X, y, train_mask, test_mask,
                                            max_iter=LR_MAX_ITER)
    print(f"Logistic Regression (raw features) test accuracy: {acc_lr:.4f}")
    print(report_lr)
    print("Saved:", pca_and_plot(X, y, "Logistic_on_raw_features"))

    emb = deepwalk_embedding(data.edge_index, data.num_nodes,
                             dimensions=config.DEEPWALK_DIM,
                             walks_per_node=config.DEEPWALK_WALKS_PER_NODE,
                             walk_length=config.DEEPWALK_WALK_LENGTH,
                             window=config.DEEPWALK_WINDOW,
                             epochs=config.DEEPWALK_EPOCHS)
    acc_dw, report_dw = train_eval_logistic(emb, y, train_mask, test_mask,
                                            max_iter=LR_MAX_ITER)
    print(f"DeepWalk + Logistic Regression test accuracy: {acc_dw:.4f}")
    print(report_dw)
    print("Saved:", pca_and_plot(emb, y, "DeepWalk_embeddings"))

    results["baselines"] = {"logistic_regression": acc_lr, "deepwalk": acc_dw}


def stage_models(dataset, data, args, results):
    print("\n=== GNN architecture comparison (multi-seed) ===")
    res = model_comparison_experiment(dataset, data, seeds=args.seeds,
                                      epochs=args.epochs)
    results["models"] = res

    # Detailed artifacts for the headline model (GraphSAGE): curves, confusion
    # matrix, t-SNE of learned embeddings.
    set_seeds(args.seeds[0])
    model = GNN("sage", dataset.num_node_features, GNN_HIDDEN,
                dataset.num_classes, num_layers=2, dropout=GNN_DROPOUT)
    single = train_node_classifier(model, data, epochs=args.epochs, lr=GNN_LR,
                                   weight_decay=GNN_WEIGHT_DECAY,
                                   patience=GNN_PATIENCE, device=DEVICE)
    print(f"GraphSAGE (seed {args.seeds[0]}): test acc {single['test_acc']:.4f}, "
          f"macro-F1 {single['test_f1_macro']:.4f}, best epoch {single['best_epoch']}")
    print("Saved:", plot_training_curves(single["history"], "GraphSAGE"))
    m = full_metrics(single["logits"], data.y.cpu(), data.test_mask.cpu())
    print("Saved:", plot_confusion_matrix(m["confusion"], "GraphSAGE_test"))
    emb = single["model"].get_embeddings(data.x.to(DEVICE),
                                         data.edge_index.to(DEVICE)).cpu().numpy()
    print("Saved:", pca_and_plot(emb, data.y.cpu().numpy(), "GraphSAGE_embeddings"))
    print("Saved:", tsne_and_plot(emb, data.y.cpu().numpy(), "GraphSAGE_embeddings_tSNE"))


def stage_aggregators(dataset, data, args, results):
    print("\n=== Aggregator comparison (paper Sec. III-B) ===")
    results["aggregators"] = aggregator_experiment(dataset, data,
                                                   seeds=args.seeds,
                                                   epochs=args.epochs)


def stage_features(dataset, data, args, results):
    print("\n=== SNA structural-feature ablation ===")
    results["features"] = feature_ablation_experiment(
        dataset, data, dataset_name=args.dataset, seeds=args.seeds,
        epochs=args.epochs)


def stage_depth(dataset, data, args, results):
    print("\n=== Depth (number of layers) experiment ===")
    results["depth"] = layer_experiment(dataset, data, layer_list=(1, 2, 3, 4),
                                        seeds=args.seeds, epochs=args.epochs)


def stage_labelrate(dataset, data, args, results):
    print("\n=== Accuracy vs. number of labeled training nodes ===")
    set_seeds(args.seeds[0])
    train_indices = torch.where(data.train_mask.cpu())[0].tolist()
    out = {}
    for p in (0.05, 0.1, 0.2, 0.4, 1.0):
        k = max(1, int(len(train_indices) * p))
        sampled = np.random.choice(train_indices, size=k, replace=False).tolist()
        d = data.clone()
        d.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool,
                                   device=data.train_mask.device)
        d.train_mask[sampled] = True
        model = GNN("sage", dataset.num_node_features, GNN_HIDDEN,
                    dataset.num_classes, num_layers=2, dropout=GNN_DROPOUT)
        res = train_node_classifier(model, d, epochs=args.epochs, lr=GNN_LR,
                                    weight_decay=GNN_WEIGHT_DECAY,
                                    patience=GNN_PATIENCE, device=DEVICE)
        out[p] = res["test_acc"]
        print(f"Proportion {p:.2f} ({k} labels) -> test acc {res['test_acc']:.4f}")
    results["labelrate"] = out


def stage_sampling(dataset, data, args, results):
    print("\n=== Neighbor sampling: fixed vs degree-adaptive ===")
    results["sampling"] = sampling_rate_experiment(
        data, fixed_size=(10, 10), num_layers=2,
        epochs=min(args.epochs, 200))


def stage_linkpred(dataset, data, args, results):
    print("\n=== Link prediction (GraphSAGE encoder + dot-product decoder) ===")
    res = run_link_prediction(data, hidden=LINKPRED_HIDDEN, out=LINKPRED_OUT,
                              epochs=LINKPRED_EPOCHS if not args.quick else 100,
                              lr=LINKPRED_LR, device=DEVICE, seed=args.seeds[0])
    print(f"Test ROC-AUC: {res['test_auc']:.4f} | Test AP: {res['test_ap']:.4f}")
    results["linkpred"] = res


def stage_community(dataset, data, args, results):
    print("\n=== Community detection (Louvain vs KMeans variants) ===")
    results["community"] = community_detection_experiment(
        dataset, data, epochs=args.epochs, make_tsne=not args.quick)


STAGE_FUNCS = {
    "baselines": stage_baselines,
    "models": stage_models,
    "aggregators": stage_aggregators,
    "features": stage_features,
    "depth": stage_depth,
    "labelrate": stage_labelrate,
    "sampling": stage_sampling,
    "linkpred": stage_linkpred,
    "community": stage_community,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=config.DATASET,
                        choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--stages", default=",".join(ALL_STAGES),
                        help="comma-separated subset of: " + ",".join(ALL_STAGES))
    parser.add_argument("--quick", action="store_true",
                        help="single seed, fewer epochs, skip t-SNE")
    args = parser.parse_args()

    args.seeds = [SEEDS[0]] if args.quick else SEEDS
    args.epochs = 100 if args.quick else GNN_EPOCHS

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGE_FUNCS]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}")

    set_seeds()
    dataset, data = load_planetoid(args.dataset)
    print(f"Dataset: {args.dataset} | nodes {data.num_nodes} | edges "
          f"{data.num_edges} | features {dataset.num_node_features} | "
          f"classes {dataset.num_classes} | device {DEVICE}")

    results = {"dataset": args.dataset, "seeds": args.seeds, "epochs": args.epochs}
    t0 = time.time()
    for s in stages:
        STAGE_FUNCS[s](dataset, data, args, results)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, f"results_{args.dataset.lower()}.json")
    merged = {}
    if os.path.exists(outfile):  # partial runs update rather than clobber
        with open(outfile) as f:
            merged = json.load(f)
    merged.update(_clean(results))
    with open(outfile, "w") as f:
        json.dump(merged, f, indent=2)
    print("Saved results to:", outfile)


if __name__ == "__main__":
    main()
