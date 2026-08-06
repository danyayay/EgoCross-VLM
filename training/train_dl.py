"""Training script for egocentric pedestrian intention prediction models.

Provides an importable :func:`train` for programmatic experiments and a CLI
for quick runs.
"""

import argparse
import copy
import datetime
import itertools
import json
import os
import yaml
import shutil
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.backbone import CLIPBackbone, CNNBackbone
from models.cross_attention_fusion import CrossAttentionFusion
from models.lstm_fusion import LSTMFusion
from models.transformer_fusion import TransformerFusion
from utils.scalers import compute_stats_from_dataloader, save_stats
from utils.util import seed_everything
from utils.vr_dataset import VRDataset, collate_fn as vr_collate


def _count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _count_total_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _tail_run_length(data) -> int:
    """Return the length of the last run of consecutive equal values in *data*."""
    counts = [len(list(group)) for _, group in itertools.groupby(data)]
    return counts[-1]


def _forward_model(model, vis, head_pose=None, gaze=None, motion=None, vel=None, goal=None):
    """Forward pass with graceful fallback for models that don't accept all inputs."""
    kwargs = {k: v for k, v in
              dict(head_pose=head_pose, gaze=gaze, motion=motion, goal=goal, vel=vel).items()
              if v is not None}
    try:
        return model(vis, **kwargs)
    except TypeError:
        kwargs.pop('vel', None)
        try:
            return model(vis, **kwargs)
        except TypeError:
            return model(vis)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, backbone: Optional[nn.Module] = None, loss_fn: Optional[nn.Module] = None):
    """Run evaluation and return (avg_loss, accuracy, class_metrics)

    class_metrics is a dict with keys 'precision','recall','f1' mapping to lists indexed by class id.
    """
    model.eval()
    # use provided loss_fn (with class weights) when available, else default
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vis, pose, gaze, motion, vel, goal, labels in loader:
            vis = vis.to(device)
            labels = labels.to(device)
            pose = pose.to(device) if pose is not None else None
            gaze = gaze.to(device) if gaze is not None else None
            motion = motion.to(device) if motion is not None else None
            vel = vel.to(device) if vel is not None else None
            goal = goal.to(device) if goal is not None else None
            if backbone is not None:
                # Only apply backbone if we received raw frames with 5 dims (B,T,H,W,3).
                # Synthetic dataset provides per-frame embeddings (B,T,feat) so skip
                # the backbone in that case.
                if vis.dim() == 5:
                    vis = backbone(vis)
            logits = _forward_model(model, vis, head_pose=pose, gaze=gaze,
                                    motion=motion, vel=vel, goal=goal)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("No samples in evaluation loader")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = float((all_preds == all_labels).sum()) / all_labels.shape[0]
    avg_loss = total_loss / total

    # determine number of classes from data
    if all_labels.size == 0:
        n_classes = 0
    else:
        n_classes = int(max(all_labels.max(), all_preds.max()) + 1)

    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    for c in range(n_classes):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        support = int((all_labels == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        support_list.append(support)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    # macro averages (unweighted)
    macro_precision = float(np.mean(precision_list)) if len(precision_list) > 0 else 0.0
    macro_recall = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0

    # micro averages (global counts)
    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    class_metrics = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "support": support_list,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "y_true": [int(x) for x in all_labels.tolist()],
        "y_pred": [int(x) for x in all_preds.tolist()],
    }
    # confusion matrix (rows=true labels, cols=predicted labels)
    try:
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(all_labels.tolist(), all_preds.tolist()):
            if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
                cm[int(t), int(p)] += 1
        class_metrics['confusion_matrix'] = cm.tolist()
    except Exception:
        # if any error occurs, still return metrics without confusion matrix
        class_metrics['confusion_matrix'] = []
    return avg_loss, accuracy, macro_f1, class_metrics


def train(
    config: Dict[str, Any],
    log_dir: Optional[str] = None,
    trial: Optional["optuna.trial.Trial"] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # timestamp for this run (used for saving model + logs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = []
    log_fh = None
    seed_everything(config.get("seed", 42))

    n_classes = int(config.get('n_classes', 5))

    def _log(msg: str):
        # print and also stream to the run log file (if available) in real time
        try:
            s = str(msg)
        except Exception:
            s = repr(msg)
        print(s)
        try:
            log_lines.append(s)
        except Exception:
            pass
        try:
            if log_fh is not None:
                log_fh.write(s + "\n")
                log_fh.flush()
        except Exception:
            pass
    
    def _save_ckpt(best_state):
        if run_dir is not None and best_state is not None:
            ckpt = os.path.join(run_dir, f"best.pt")
            torch.save({"state_dict": best_state, "config": config, "timestamp": timestamp}, ckpt)
            _log(f"  Saved best model to {ckpt}")

    def _plot_training_curve(train_losses, val_losses):
        import matplotlib.pyplot as plt
        if run_dir is None:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Curve')
        plt.legend()
        plot_path = os.path.join(run_dir, 'training_curve.png')
        plt.savefig(plot_path)
        plt.close()

    # determine run directory for artifacts. If the caller sets
    # config['_no_timestamp_dir']=True then we will use the provided
    # log_dir as-is (useful when run by Optuna where trial dirs are
    # already timestamped). Otherwise create a timestamped subfolder
    # under log_dir (when provided).
    run_dir = None
    if log_dir is not None:
        if config.get('_no_timestamp_dir', False):
            run_dir = log_dir
        else:
            run_dir = os.path.join(log_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        # save the config for reproducibility
        try:
            with open(os.path.join(run_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        except Exception:
            pass
        # open a streaming log file so _log can write in real time
        try:
            log_path = os.path.join(run_dir, 'training_log.txt')
            # open in append mode so repeated runs don't clobber previous logs
            log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)
        except Exception:
            log_fh = None
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        _log(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")
        _log(f"Memory Limit Set: {(total_memory * 0.8) / 1e9:.2f} GB")

    _log("Starting training with config:")
    _log(json.dumps(config, separators=(",", ":"), sort_keys=True))

    # visual_dim is flattened frame size (H*W*3)
    visual_encoder = config.get("visual_encoder", None)
    if visual_encoder == "pixels":
        frame_size = tuple(config.get("frame_size", (32, 32)))
        visual_dim = frame_size[0] * frame_size[1] * 3
    elif visual_encoder == 'cnn':
        visual_dim = config.get("backbone_out_dim", 256)
    elif visual_encoder == 'clip':
        # CLIP output dim depends on model; default ViT-B/32 has 512-dim embeddings
        visual_dim = config.get("backbone_out_dim", 512)
    elif visual_encoder is None:
        visual_dim = 0

    # dataset selection and visual encoder
    dataset_name = config.get("dataset_name", "synthetic")
    if dataset_name == "vr":
        train_ds = VRDataset(
            annotations_path=config["anno_train_path"],
            videodata_dir=config["videodata_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            visual_dim=visual_dim,
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            vel_cols=config.get("vel_cols", None),
            goal_cols=config.get("goal_cols", None),
            crop_anchor=config.get("crop_anchor", None),
            visual_encoder=visual_encoder,
            return_raw_frames=(visual_encoder != "pixels")
        )
        val_ds = VRDataset(
            annotations_path=config["anno_val_path"],
            videodata_dir=config["videodata_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            visual_dim=visual_dim,
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            vel_cols=config.get("vel_cols", None),
            goal_cols=config.get("goal_cols", None),
            crop_anchor=config.get("crop_anchor", None),
            visual_encoder=visual_encoder,
            return_raw_frames=(visual_encoder != "pixels")
        )
        test_ds = VRDataset(
            annotations_path=config["anno_test_path"],
            videodata_dir=config["videodata_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            visual_dim=visual_dim,
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            vel_cols=config.get("vel_cols", None),
            goal_cols=config.get("goal_cols", None),
            crop_anchor=config.get("crop_anchor", None),
            visual_encoder=visual_encoder,
            return_raw_frames=(visual_encoder != "pixels")
        )
        collate = vr_collate
        # infer per-modality dims from the first training sample
        pose_dim = config.get("pose_dim", train_ds[0][1].shape[1] if train_ds[0][1].numel() > 0 else 0)
        gaze_dim = config.get("gaze_dim", train_ds[0][2].shape[1] if train_ds[0][2].numel() > 0 else 0)
        motion_dim = config.get("motion_dim", train_ds[0][3].shape[1] if train_ds[0][3].numel() > 0 else 0)
        vel_dim = config.get("vel_dim", train_ds[0][4].shape[1] if train_ds[0][4].numel() > 0 else 0)
        goal_dim = config.get("goal_dim", train_ds[0][5].shape[1] if train_ds[0][5].numel() > 0 else 0)
    
    # data loaders
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 64), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 64), shuffle=False, collate_fn=collate)

    # Fit per-modality scalers on the training set and save them. This avoids data leakage
    # since we compute statistics only from train_loader. The returned stats is a dict
    # like {'pose': {'mean':..., 'std':...}, 'gaze':..., 'motion':..., 'goal_dist':...}
    log_dir = log_dir or config.get("log_dir", None)
    try:
        if config.get("egocentric_only", False):
            stats = {}
        else: # with extra modalities, compute and save stats as usual
            stats = compute_stats_from_dataloader(train_loader)
            if run_dir is not None:
                save_stats(stats, run_dir)

            # helper to attach stats to dataset or Subset.dataset
            def _attach_stats(ds_obj, stats_dict):
                base = ds_obj.dataset if isinstance(ds_obj, Subset) else ds_obj
                if 'pose' in stats_dict:
                    setattr(base, 'pose_scaler', stats_dict['pose'])
                if 'gaze' in stats_dict:
                    setattr(base, 'gaze_scaler', stats_dict['gaze'])
                if 'motion' in stats_dict:
                    setattr(base, 'motion_scaler', stats_dict['motion'])
                if 'vel' in stats_dict:
                    setattr(base, 'vel_scaler', stats_dict['vel'])
                if 'goal_dist' in stats_dict:
                    setattr(base, 'goal_dist_scaler', stats_dict['goal_dist'])
            _log(f"stats={stats}")
            _attach_stats(train_ds, stats)
            _attach_stats(val_ds, stats)
            _attach_stats(test_ds, stats)
    except Exception as e:
        _log(f"Warning: could not compute/save scalers: {e}")
        # if stats computation fails, continue without scalers
        stats = {}

    # model selection
    model_type = config.get("model", "transformer")
    if model_type == "transformer":
        model = TransformerFusion(
            visual_dim=visual_dim,
            pose_dim=pose_dim,
            gaze_dim=gaze_dim,
            motion_dim=motion_dim,
            vel_dim=vel_dim,
            goal_dim=goal_dim,
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.0),
            n_classes=n_classes,
            max_time_steps=config.get("max_time_steps", 64),
        ).to(device)
    elif model_type == "cross_attention":
        model = CrossAttentionFusion(
            visual_dim=visual_dim,
            pose_dim=pose_dim,
            gaze_dim=gaze_dim,
            motion_dim=motion_dim,
            vel_dim=vel_dim,
            goal_dim=goal_dim,
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 3),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1),
            n_classes=n_classes,
            max_time_steps=config.get("max_time_steps", 64),
            predict_future_steps=config.get("predict_future_steps", 0),
        ).to(device)
    elif model_type == "lstm":
        model = LSTMFusion(
            visual_dim=visual_dim,
            pose_dim=pose_dim,
            gaze_dim=gaze_dim,
            motion_dim=motion_dim,
            vel_dim=vel_dim,
            goal_dim=goal_dim,
            d_model=config.get("d_model", 256),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            n_classes=n_classes,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose from 'transformer', 'cross_attention', 'lstm'.")
    _log(model)

    # Print parameter counts for the model (trainable vs total) before training
    try:
        model_trainable = _count_params(model)
        model_total = _count_total_params(model)
        _log(f"Model parameters: {model_trainable} trainable / {model_total} total")
    except Exception:
        _log("Model parameter counting failed")

    # optional backbone for raw frames: SimpleCNN or CLIP
    backbone = None
    if visual_encoder == "cnn":
        backbone = CNNBackbone(in_ch=3, out_dim=config.get("backbone_out_dim", 256), hidden=config.get("backbone_hidden", 64)).to(device)
    elif visual_encoder == "clip":
        backbone = CLIPBackbone(model_name=config.get("clip_model", "ViT-B/32"), device=device, freeze=config.get("freeze_clip", False))

    # Log backbone and combined parameter counts (if backbone exists)
    try:
        if backbone is not None:
            backbone_trainable = _count_params(backbone)
            backbone_total = _count_total_params(backbone)
            model_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            combined_trainable = model_trainable + backbone_trainable
            _log(f"Backbone parameters: {backbone_trainable} trainable / {backbone_total} total")
            _log(f"Combined trainable parameters (model + backbone): {combined_trainable}")
    except Exception:
        _log("Backbone parameter counting failed")

    if backbone is not None:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(backbone.parameters()), lr=config.get("lr", 1e-4))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # compute class weights from the training set to handle class imbalance if required
    if config.get('is_weighted_sampler', True):
        weights = train_ds.get_label_weights_basedon_count(n_classes=n_classes)
    else:
        weights = np.ones(n_classes, dtype=np.float32)

    _log(f"weights: {weights}")
    _log(f'Labeling map: {train_ds.label_map}' )
    weights = torch.tensor(weights, dtype=torch.float32).to(device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # track bests for multiple metrics but monitor a single chosen eval metric
    best_val_acc = 0.0
    best_val_macrof1 = 0.0
    best_epoch = 0
    best_state = None
    # the metric to monitor for model selection / early stopping: 'acc', 'macro_f1' or 'loss'
    monitored_metric = config.get('monitored_metric', 'acc')
    # monitoring mode: 'max' for accuracy/F1, 'min' for loss
    if monitored_metric in ("acc", "macro_f1"):
        monitor_mode = 'max'
        best_monitored_value = float('-inf')
    elif monitored_metric == 'loss':
        monitor_mode = 'min'
        best_monitored_value = float('inf')
    else:
        raise ValueError(f"Unknown monitored metric: {monitored_metric}")

    epochs = int(config.get("epochs", 3))
    no_improvement_patience = int(config.get('early_stop_patience', 20))
    epochs_since_improve = 0
    # additional stopper: if neither val_acc nor val_macrof1 change for this many epochs, stop
    no_change_patience = no_improvement_patience // 2 + 1
    train_loss_list, val_loss_list, val_monitor_list = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for vis, pose, gaze, motion, vel, goal, labels in tqdm(train_loader):
            vis = vis.to(device) if vis is not None else None
            labels = labels.to(device)
            pose = pose.to(device) if pose is not None else None
            gaze = gaze.to(device) if gaze is not None else None
            motion = motion.to(device) if motion is not None else None
            vel = vel.to(device) if vel is not None else None
            goal = goal.to(device) if goal is not None else None

            optimizer.zero_grad()
            vis_emb = backbone(vis) if (backbone is not None and vis.dim() == 5) else vis
            logits = _forward_model(model, vis_emb, head_pose=pose, gaze=gaze,
                                    motion=motion, vel=vel, goal=goal)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_samples += labels.size(0)

        scheduler.step()
        train_loss = running_loss / running_samples

        val_loss, val_acc, val_macrof1, val_metrics = evaluate(model, val_loader, device, backbone=backbone, loss_fn=loss_fn)
        _log(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macrof1={val_macrof1:.4f}, lr={scheduler.get_last_lr()[0]}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # always update secondary tracked metrics for reporting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_macrof1 > best_val_macrof1:
            best_val_macrof1 = val_macrof1

        # update best according to the chosen monitored metric (respect min/max)
        current_monitored = {"acc": val_acc, "macrof1": val_macrof1, "loss": val_loss}.get(monitored_metric, None)
        val_monitor_list.append(current_monitored)
        is_better = (current_monitored > best_monitored_value) if monitor_mode == 'max' else (current_monitored < best_monitored_value)
        if is_better:
            best_monitored_value = current_monitored
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _save_ckpt(best_state)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        
        # update no-change counter
        epochs_since_no_change = _tail_run_length(val_monitor_list)
        epochs_since_no_change = 0 if epochs_since_no_change == 1 else epochs_since_no_change
        _log(f"  epochs_since_improve={epochs_since_improve} epochs_since_no_change={epochs_since_no_change}")

        if epochs_since_improve >= no_improvement_patience:
            _log(f"Early stopping: no improvement for {no_improvement_patience} epochs (stopping at epoch {epoch})")
            break
        if epochs_since_no_change >= no_change_patience:
            _log(f"Early stopping: no change in val_acc or val_macrof1 for {no_change_patience} epochs (stopping at epoch {epoch})")
            break

        # --- INSERT PRUNING LOGIC HERE ---
        if trial is not None:
            trial.report(current_monitored, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    _log(f"\nFinished training. \nbest_val_{monitored_metric}= {best_monitored_value:.4f} at epoch {best_epoch}")
    _plot_training_curve(train_loss_list, val_loss_list)

    # Evaluate on test set. If we have a saved best_state, load it for final evaluation.
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            # in case of mismatch, ignore and use final weights
            pass

    # evaluate test metrics
    val_loss, val_acc, val_macrof1, val_metrics = evaluate(model, val_loader, device, backbone=backbone, loss_fn=loss_fn)
    test_loss, test_acc, test_macrof1, test_metrics = evaluate(model, test_loader, device, backbone=backbone, loss_fn=loss_fn)
    _log(f"\nTest: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_macrof1={test_macrof1:.4f}")

    # Nicely print per-class metrics
    precs = test_metrics.get("precision", [])
    recs = test_metrics.get("recall", [])
    f1s = test_metrics.get("f1", [])
    supports = test_metrics.get("support", [])
    ncls = len(precs)
    if ncls > 0:
        header = f"{'class':>5} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
        _log("\nPer-class metrics:")
        _log(header)
        _log("-" * len(header))
        for c in range(ncls):
            _log(f"{c:5d} {precs[c]:10.4f} {recs[c]:10.4f} {f1s[c]:10.4f} {supports[c]:8d}")
        # macro & micro
        macro = test_metrics.get("macro", {})
        micro = test_metrics.get("micro", {})
        _log("\nAverages:")
        _log(f"macro - precision: {macro.get('precision',0.0):.4f}, recall: {macro.get('recall',0.0):.4f}, f1: {macro.get('f1',0.0):.4f}")
        _log(f"micro - precision: {micro.get('precision',0.0):.4f}, recall: {micro.get('recall',0.0):.4f}, f1: {micro.get('f1',0.0):.4f}")

    # save test metrics to run_dir if provided
    if run_dir is not None:
        metrics_path = os.path.join(run_dir, "test_metrics.txt")
        try:
            with open(metrics_path, "w") as f:
                f.write(f"test_loss {test_loss}\n")
                f.write(f"test_acc {test_acc}\n")
                f.write("class,precision,recall,f1\n")
                ncls = len(test_metrics.get("precision", []))
                for c in range(ncls):
                    p = test_metrics["precision"][c]
                    r = test_metrics["recall"][c]
                    f1 = test_metrics["f1"][c]
                    f.write(f"{c},{p:.6f},{r:.6f},{f1:.6f}\n")
            # also save confusion matrix as CSV if available
            try:
                cm = test_metrics.get('confusion_matrix', [])
                if cm and len(cm) > 0:
                    import csv
                    cm_path = os.path.join(run_dir, 'confusion_matrix.csv')
                    with open(cm_path, 'w', newline='') as cf:
                        writer = csv.writer(cf)
                        # header: predicted cols
                        writer.writerow(['true/pred'] + [str(i) for i in range(len(cm))])
                        for i, row in enumerate(cm):
                            writer.writerow([str(i)] + row)
                    label_map = getattr(test_ds, "label_map", None) or {}
                    id_to_label = {int(v): str(k) for k, v in label_map.items()}
                    labels = [id_to_label.get(i, str(i)) for i in range(len(cm))]
                    normalized = []
                    for row in cm:
                        denom = float(sum(row))
                        normalized.append([float(v) / denom if denom else 0.0 for v in row])
                    with open(os.path.join(run_dir, "confusion_matrix_labeled.json"), "w") as lf:
                        json.dump({"labels": labels, "rows": "true", "columns": "predicted",
                                   "counts": cm, "row_normalized": normalized}, lf, indent=2)
            except Exception:
                pass
            # The unshuffled test loader preserves test_ds.samples order.
            try:
                y_true = test_metrics.get("y_true", [])
                y_pred = test_metrics.get("y_pred", [])
                label_map = getattr(test_ds, "label_map", None) or {}
                id_to_label = {int(v): str(k) for k, v in label_map.items()}
                samples = getattr(test_ds, "samples", [])
                predictions = []
                for i, (truth, pred) in enumerate(zip(y_true, y_pred)):
                    sample = samples[i] if i < len(samples) else {}
                    predictions.append({
                        "video_id": sample.get("video_id", str(i)),
                        "answer": id_to_label.get(int(truth), str(truth)),
                        "pred_answer": id_to_label.get(int(pred), str(pred)),
                    })
                with open(os.path.join(run_dir, "test_results.json"), "w") as pf:
                    json.dump(predictions, pf, indent=2)
                with open(os.path.join(run_dir, "test_metrics.json"), "w") as mf:
                    json.dump({"seed": int(config.get("seed", 42)),
                               "test_loss": test_loss, "test_acc": test_acc,
                               "test_macro_f1": test_macrof1,
                               "metrics": test_metrics}, mf, indent=2)
            except Exception as exc:
                _log(f"Warning: could not save structured test artifacts: {exc}")
        except Exception:
            pass

    # save training log
    if run_dir is not None:
        try:
            # if we have an open streaming log file, close it before writing the final copy
            try:
                if log_fh is not None:
                    log_fh.close()
            except Exception:
                pass
            log_path = os.path.join(run_dir, f"training_log.txt")
            with open(log_path, "w") as f:
                for l in log_lines:
                    f.write(str(l) + "\n")
            _log(f"Saved training log to {log_path}")
        except Exception:
            pass

    return {
        "best_val_acc": best_val_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macrof1": val_macrof1,
        "val_metrics": val_metrics,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macrof1": test_macrof1,
        "test_metrics": test_metrics,
        # include monitored metric and its name so callers (e.g., Optuna) can use it
        "monitored_metric": monitored_metric,
        "best_monitored": best_monitored_value,
        "config": config,
    }


def run_optuna(base_cfg: Dict[str, Any]) -> Optional[object]:
    """Run an Optuna study using the training function as the objective.

    Each trial runs train(...) for a small number of epochs in a temporary
    work directory. The best trial params are written to base_cfg['log_dir']
    (if provided).
    """
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install it with 'pip install optuna' to use --hp-tune.")

    # create a timestamped study directory to collect all artifacts
    study_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_work = base_cfg.get('log_dir') or os.getcwd()
    study_dir = os.path.join(base_work, f"{study_ts}_optuna")
    os.makedirs(study_dir, exist_ok=True)
    # streaming optuna log
    optuna_log_fh = None
    try:
        optuna_log_path = os.path.join(study_dir, 'optuna_log.txt')
        optuna_log_fh = open(optuna_log_path, 'a', encoding='utf-8', buffering=1)
    except Exception:
        optuna_log_fh = None

    def _optuna_log(msg: str):
        try:
            s = str(msg)
        except Exception:
            s = repr(msg)
        print(s)
        try:
            if optuna_log_fh is not None:
                optuna_log_fh.write(s + "\n")
                optuna_log_fh.flush()
        except Exception:
            pass

    def _remove_log_file(log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Removed log file: {log_dir}")


    def _objective(trial) -> float:
        cfg = dict(base_cfg)
        # sample hyperparameters
        cfg['lr'] = trial.suggest_float('lr', 1e-5, 1e-3)
        cfg['batch_size'] = int(trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]))

        cfg['num_layers'] = int(trial.suggest_int('num_layers', 1, 4))
        cfg['visual_encoder'] = base_cfg.get('visual_encoder')
        if base_cfg.get('visual_encoder'):
            # cfg['d_model'] = int(trial.suggest_int('d_model', 32, 256))
            cfg['d_model'] = int(trial.suggest_categorical('d_model', [32, 64, 96, 128, 160, 192, 224, 256]))
            cfg['T'] = trial.suggest_categorical('T', [8, 16, 30, 60])
        # cfg['model'] = trial.suggest_categorical('model', ['concat', 'egocognav', 'hybrid'])
        cfg['model'] = base_cfg.get('model')
        
        if base_cfg.get('model') in ['transformer', 'cross_attention']:
            cfg['nhead'] = int(trial.suggest_categorical('nhead', [1, 2, 4, 8, 16]))
            ff_choice = trial.suggest_categorical('ff_choice', ['d2', 'd4', '512'])
            if ff_choice == '512':
                cfg['dim_feedforward'] = 512
            elif ff_choice == 'd2':
                cfg['dim_feedforward'] = int(cfg['d_model'] * 2)
            else:
                cfg['dim_feedforward'] = int(cfg['d_model'] * 4)
            cfg['dropout'] = float(trial.suggest_float('dropout', 0.0, 0.3))
        elif base_cfg.get('model') == 'lstm':
            cfg['hidden_dim'] = int(trial.suggest_int('hidden_dim', 32, 256))

        if base_cfg.get('visual_encoder'):
            if base_cfg.get('visual_encoder') != 'clip':
                resize_size = int(trial.suggest_categorical('resize_size', [32, 64, 128, 224]))
            else:
                resize_size = 224  # CLIP default
            cfg['frame_size'] = (resize_size, resize_size)

        if base_cfg.get('visual_encoder') == 'pixels' or base_cfg.get('visual_encoder') is None:
            cfg['backbone_out_dim'] = None
            cfg['backbone_hidden'] = None
            cfg['clip_model'] = None
            cfg['freeze_clip'] = False
        elif base_cfg.get('visual_encoder') == 'cnn':
            cfg['backbone_out_dim'] = int(trial.suggest_int('backbone_out_dim', 64, 256))
            cfg['backbone_hidden'] = int(trial.suggest_int('backbone_hidden', 32, 128))
            cfg['clip_model'] = None
            cfg['freeze_clip'] = False
        elif base_cfg.get('visual_encoder') == 'clip':
            cfg['clip_model'] = base_cfg.get('clip_model')
            cfg['freeze_clip'] = base_cfg.get('freeze_clip')
            cfg['backbone_out_dim'] = 512  # assuming ViT-B/32
            cfg['backbone_hidden'] = int(trial.suggest_int('backbone_hidden', 32, 128))

        # create a per-trial directory under the study dir so that all logs
        # and checkpoints from trials are retained in a single timestamped
        # folder. We set a flag so train() will not create an extra timestamp
        # subfolder.
        trial_dir = os.path.join(study_dir, 'trials', f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        cfg['_no_timestamp_dir'] = True

        max_attempts = 2
        error = None
        for _ in range(max_attempts):
            try:
                result = train(cfg, log_dir=trial_dir, trial=trial)
                # capture metrics for Optuna reporting and later analysis
                # prefer the new 'best_monitored' key (monitored metric), fall back to best_val_acc
                val_score = float(result.get('best_monitored', -1))
                # attach these to trial user attrs
                # record the metric that was monitored for this trial
                trial.set_user_attr('monitored_metric', result.get('monitored_metric', -1))
                trial.set_user_attr('val_acc', float(result.get('val_acc', -1)))
                trial.set_user_attr('val_macrof1', result.get('val_macrof1', -1))
                trial.set_user_attr('test_acc', float(result.get('test_acc', -1)))
                trial.set_user_attr('test_macrof1', result.get('test_macrof1', -1))
                
                return val_score
            except optuna.TrialPruned:
                raise 
            except Exception as e:  # pragma: no cover - avoid crashing the study
                _optuna_log(f"Trial crashed: {e}, Retrying...")
                error = e
                val_score = -1
                continue  # retry

        _remove_log_file(trial_dir)
        raise TrialPruned(f'Trial failed after {max_attempts} attempts because {error}')

    # use a sqlite storage so the study can be inspected with optuna-dashboard
    
    def get_feat_indicators(cfg):
        feats_indicators = []
        feats_indicators.append('1') if cfg.get('motion_cols', []) != [] else feats_indicators.append('0')
        feats_indicators.append('1') if cfg.get('pose_cols', []) != [] else feats_indicators.append('0')
        feats_indicators.append('1') if cfg.get('gaze_cols', []) != [] else feats_indicators.append('0')
        feats_indicators.append('1') if cfg.get('vel_cols', []) != [] else feats_indicators.append('0')
        feats_indicators.append('1') if cfg.get('goal_cols', []) != [] else feats_indicators.append('0')
        return ''.join(feats_indicators)
    
    monitored = base_cfg.get('monitored_metric', 'acc')
    if monitored in ['acc', 'macro_f1']:
        direction = 'maximize'
    elif monitored in ['loss']:
        direction = 'minimize'

    my_pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study_name = f'egocentric_{base_cfg.get("cls_type")}_{base_cfg.get("visual_encoder", "none")}_{base_cfg.get("model")}_{get_feat_indicators(base_cfg)}_{base_cfg.get("crop_anchor", "padding")}'
    optuna_url = base_cfg.get('optuna_url', "sqlite:///db.sqlite_optuna")
    study = optuna.create_study(
        storage=optuna_url,
        load_if_exists=True,
        direction=direction,
        pruner=my_pruner,
        study_name=study_name
    )
    print(f"Optuna study created. Study name: {study.study_name}, storage: {optuna_url}")
    # with good initial params
    study.enqueue_trial(base_cfg)

    hp_trials = base_cfg.get('hp_trials', 100)
    _optuna_log(f"Starting Optuna study optimization with n_trials={hp_trials}")
    study.optimize(_objective, n_trials=hp_trials, catch=())  # catch all exceptions to avoid study crashes

    # save best params and trials into study_dir
    best = study.best_trial
    try:
        best_entry = {'value': best.value, 'params': best.params}
        # include any user attrs recorded during the objective
        try:
            best_entry['user_attrs'] = best.user_attrs
        except Exception:
            best_entry['user_attrs'] = {}
        with open(os.path.join(study_dir, 'optuna_best_params.json'), 'w') as f:
            json.dump(best_entry, f, indent=4)
    except Exception:
        pass

    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(study_dir, 'optuna_trials.csv'), index=False)
    except Exception:
        # if pandas not available or conversion fails, fallback to writing simple text
        try:
            with open(os.path.join(study_dir, 'optuna_trials.txt'), 'w') as f:
                for t in study.trials:
                    f.write(str({'number': t.number, 'value': t.value, 'params': t.params}) + "\n")
        except Exception:
            pass

    # write a detailed per-trial CSV that includes user_attrs (val_macrof1, test metrics)
    try:
        import csv
        detailed_path = os.path.join(study_dir, 'optuna_trials_detailed.csv')
        with open(detailed_path, 'w', newline='') as csvfile:
            fieldnames = ['number', 'value', 'params', 'monitored_metric', 'val_monitored', 'val_macrof1', 'test_acc', 'test_macrof1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for t in study.trials:
                ua = getattr(t, 'user_attrs', {}) if hasattr(t, 'user_attrs') else {}
                row = {
                    'number': t.number,
                    'value': t.value,
                        'params': t.params,
                        'monitored_metric': ua.get('monitored_metric', ''),
                        'val_monitored': ua.get('val_monitored', ''),
                        'val_macrof1': ua.get('val_macrof1', ''),
                        'test_acc': ua.get('test_acc', ''),
                        'test_macrof1': ua.get('test_macrof1', ''),
                }
                writer.writerow(row)
    except Exception:
        pass

    # write a human-readable summary
    try:
        summary_path = os.path.join(study_dir, 'optuna_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Optuna study summary - best value: {best.value}\n")
            f.write(f"Best params: {best.params}\n\n")
            f.write("Trials:\n")
            for t in study.trials:
                f.write(f"#%d value={t.value} params={t.params}\n" % t.number)
    except Exception:
        pass

    _optuna_log(f"Optuna study finished. Best value: {best.value}, params: {best.params}. Artifacts in {study_dir}")
    try:
        if optuna_log_fh is not None:
            optuna_log_fh.close()
    except Exception:
        pass
    return study


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/tune_dl.yaml', help="Path to the config YAML file")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed")
    parser.add_argument("--log-dir", default=None, help="Override the configured output root")
    return parser.parse_args()


def format_config(raw_cfg):
    cfg = {}
    
    # Flatten the configuration structure
    if 'training' in raw_cfg:
        cfg.update(raw_cfg['training'])
    if 'data' in raw_cfg:
        cfg.update(raw_cfg['data'])
    if 'tune' in raw_cfg:
        cfg.update(raw_cfg['tune'])
        
    if 'model_name' in raw_cfg:
        cfg['model'] = raw_cfg['model_name']
        cfg.update(raw_cfg['models'].get(raw_cfg['model_name'], {}))
    
    if 'visual_encoder' in raw_cfg:
        cfg['visual_encoder'] = raw_cfg['visual_encoder']
        cfg.update(raw_cfg['backbone_models'].get(raw_cfg['visual_encoder'], {}))

    cfg['task'] = raw_cfg.get('task', 'train')
    cfg['log_dir'] = raw_cfg.get('log_dir', 'logs/train_dl')

    cfg["anno_train_path"] = os.path.join(cfg.get("anno_dir", ""), cfg.get("anno_train_filename", ""))
    cfg["anno_val_path"] = os.path.join(cfg.get("anno_dir", ""), cfg.get("anno_val_filename", ""))
    cfg["anno_test_path"] = os.path.join(cfg.get("anno_dir", ""), cfg.get("anno_test_filename", ""))

    # double check correctness 
    if cfg.get("egocentric_only", False):
        print("Extra modalities disabled; using only egocentric video.")
        cfg["pose_cols"] = []
        cfg["gaze_cols"] = []
        cfg["motion_cols"] = []
        cfg["vel_cols"] = []
        cfg["goal_cols"] = []
    
    return cfg 


if __name__ == "__main__":
    args = parse_args()
    
    with open(args.config, 'r') as f:
        raw_cfg = yaml.safe_load(f)
        
    cfg = format_config(raw_cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir

    if cfg.get("task") == "tune":
        # run Optuna tuning
        study = run_optuna(cfg)
        # after tuning, print best and exit
        if study is not None:
            best = study.best_trial
            print("Optuna best:", best.number, best.value, best.params)
        else:
            print("Optuna did not run (optuna missing?)")
    else:
        result = train(cfg, log_dir=cfg.get("log_dir"))
