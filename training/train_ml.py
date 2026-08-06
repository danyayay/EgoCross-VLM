#!/usr/bin/env python3
"""
ml_train.py

Full prediction pipeline for classification from time-series features stored in a .npz file.

Features expected in the .npz file:
 - common keys: 'feat' (np.array), optional 'labels' (np.array) or other label keys

The script supports: RandomForest, XGBoost (if installed), 1D-CNN, LSTM and GRU (PyTorch).

It will:
 - load features and labels
 - split train/test
 - normalize features (optional)
 - train selected models
 - evaluate and save metrics, confusion matrices and plots

Example:
  python3 ml_train.py --npz features/ml/feats_bin.npz --label-key labels --models rf xgb cnn lstm --out-dir results/ml_run

"""
import argparse
import json
import re
import os
import random
import shutil
from pathlib import Path

import numpy as np

# plotting imports (used later, but safe to import)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


from utils.vr_dataset import split_person_groups


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz(path: str, label_key: str):
    data = np.load(path, allow_pickle=True)

    # expectations: data['feat'] -> features, data['feat_id'] -> identifiers (strings or structured)
    if 'feat' not in data:
        raise KeyError("expected key 'feat' in npz")
    X = data['feat']
    print(f"Loaded features X with shape {X.shape} from {path}")

    feat_id = None
    if 'feat_id' in data:
        feat_id = data['feat_id']

    # label_key may be an index into a structured feat_id or the name of a key in the archive
    y = data['feat_id'][:, label_key]

    return X, y, feat_id


def parse_person_from_feat_id(fid):
    """Extract person id like 'P10' from a feat_id entry.
    fid may be bytes, tuple, or string. Return string person id or the original fid if not found.
    """
    if fid is None:
        return None
    if isinstance(fid, bytes):
        try:
            fid = fid.decode('utf-8')
        except Exception:
            fid = str(fid)
    if not isinstance(fid, str):
        try:
            # try to stringify (e.g., numpy bytes)
            fid = str(fid)
        except Exception:
            return None
    m = re.search(r'(P\d+)', fid)
    if m:
        return m.group(1)
    # try alternate pattern 'p\d+'
    m = re.search(r'(p\d+)', fid)
    if m:
        return m.group(1).upper()
    return None


# def split_person_groups(feat_ids, seed=42, ratios=(6, 1, 3)):
#     """Split persons into train/val/test groups by person-level splitting with given ratios.
#     Returns dict with keys 'train','val','test' -> index lists and 'persons' lists.
#     """
#     # extract person id per sample
#     persons_per_sample = [parse_person_from_feat_id(x) for x in feat_ids]
#     unique_persons = sorted([p for p in set(persons_per_sample) if p is not None])
#     if len(unique_persons) == 0:
#         raise ValueError('No person ids found in feat_id to perform group split')
#     rng = np.random.RandomState(seed)
#     rng.shuffle(unique_persons)

#     total = sum(ratios)
#     n = len(unique_persons)
#     n_train = max(1, int(n * (ratios[0] / total)))
#     n_val = max(1, int(n * (ratios[1] / total)))
#     # ensure sum <= n
#     if n_train + n_val >= n:
#         n_train = max(1, n - 2)
#         n_val = 1
#     n_test = n - n_train - n_val

#     train_persons = unique_persons[:n_train]
#     val_persons = unique_persons[n_train:n_train + n_val]
#     test_persons = unique_persons[n_train + n_val:]
#     print('Split persons:')
#     print(f'  train ({len(train_persons)}): {train_persons}')
#     print(f'  val   ({len(val_persons)}): {val_persons}')
#     print(f'  test  ({len(test_persons)}): {test_persons}')

#     idx_train = [i for i, p in enumerate(persons_per_sample) if p in train_persons]
#     idx_val = [i for i, p in enumerate(persons_per_sample) if p in val_persons]
#     idx_test = [i for i, p in enumerate(persons_per_sample) if p in test_persons]

#     return {
#         'train_idx': np.array(idx_train, dtype=int),
#         'val_idx': np.array(idx_val, dtype=int),
#         'test_idx': np.array(idx_test, dtype=int),
#         'train_persons': train_persons,
#         'val_persons': val_persons,
#         'test_persons': test_persons,
#     }


def metrics_report(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for i, lab in enumerate(labels if labels is not None else range(len(p))):
        per_class.append({'label': int(lab), 'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f1[i]), 'support': int(support[i])})
    # compute macro f1
    try:
        macro_f1 = float(np.mean(f1))
    except Exception:
        macro_f1 = None
    return {'accuracy': float(acc), 'macro_f1': macro_f1, 'per_class': per_class, 'confusion': cm.tolist()}


def plot_confusion(cm, labels, outpath: str, fmt='png', model_name: str = None, accuracy: float = None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(cm), annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title = 'Confusion Matrix'
    if model_name is not None:
        title = f"{title} - {model_name}"
    if accuracy is not None:
        try:
            title = f"{title} (acc={accuracy:.3f})"
        except Exception:
            title = f"{title} (acc={accuracy})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath + f'.{fmt}')
    plt.close()


def plot_per_class_metrics(per_class: list, le_class: list, outpath: str, fmt='png', model_name: str = None, accuracy: float = None):
    # per_class: list of dicts with label, precision, recall, f1, support
    labels = [str(x['label']) for x in per_class]
    precision = [x['precision'] for x in per_class]
    recall = [x['recall'] for x in per_class]
    f1 = [x['f1'] for x in per_class]

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label='precision')
    plt.bar(x, recall, width, label='recall')
    plt.bar(x + width, f1, width, label='f1')
    plt.xticks(x, le_class, rotation=45)
    plt.ylabel('score')
    title = 'Per-class metrics'
    if model_name is not None:
        title = f"{title} - {model_name}"
    if accuracy is not None:
        try:
            title = f"{title} (acc={accuracy:.3f})"
        except Exception:
            title = f"{title} (acc={accuracy})"
    plt.title(title)
    # also annotate accuracy on the figure for quick visibility
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + f'.{fmt}')
    plt.close()


def plot_loss(history: dict, outpath: str, fmt='png', model_name: str = None):
    """Plot training and (optional) validation loss per epoch.
    history: {'train_loss': [...], 'val_loss': [...] (optional)}
    """
    if history is None:
        return
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', None)
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label='train_loss')
    if val_loss is not None and len(val_loss) == len(train_loss):
        plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    title = 'Training loss'
    if model_name:
        title = f"{title} - {model_name}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + f'.{fmt}')
    plt.close()


def compute_hyperparam_importance(trials: list):
    """Compute a simple hyperparameter importance score from tuning trials.

    trials: list of {'params': {...}, 'val_acc': float}
    Returns a dict with raw std-of-means per hyperparam, normalized importances, and per-value means.
    """
    if not trials:
        return {}
    # collect all hyperparameter names
    param_names = set()
    for t in trials:
        param_names.update(t.get('params', {}).keys())
    values = {}
    importances_raw = {}
    for p in param_names:
        # collect accs per param value
        val_to_accs = {}
        for t in trials:
            v = t.get('params', {}).get(p, None)
            key = str(v)
            val_to_accs.setdefault(key, []).append(float(t.get('val_acc', 0.0)))
        # compute mean acc per value
        means = {k: float(np.mean(vv)) for k, vv in val_to_accs.items()}
        values[p] = means
        # importance as std of these means
        if len(means) > 1:
            importances_raw[p] = float(np.std(list(means.values())))
        else:
            importances_raw[p] = 0.0
    # normalize
    total = sum(importances_raw.values())
    if total > 0:
        importances_norm = {k: float(v / total) for k, v in importances_raw.items()}
    else:
        importances_norm = {k: 0.0 for k in importances_raw}
    return {'raw': importances_raw, 'norm': importances_norm, 'values': values}


def train_rf(X_train, y_train, X_test, y_test, args):
    if getattr(args, 'class_balanced', False):
        print('Using class-balanced weights for RandomForestClassifier')
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=args.n_jobs, class_weight='balanced')
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=args.n_jobs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def train_xgb(X_train, y_train, X_test, y_test, args):
    if not HAS_XGB:
        raise ImportError('xgboost not installed')
    # For binary tasks we can set scale_pos_weight; for multiclass pass per-sample weights
    if getattr(args, 'class_balanced', False) and getattr(args, 'qtype', 'binary') == 'binary':
        # compute ratio of negative/positive (assume two classes)
        try:
            classes, counts = np.unique(y_train, return_counts=True)
            if len(classes) == 2:
                # positive class we'll assume is classes[1]
                neg = int(counts[0])
                pos = int(counts[1])
                scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
            else:
                scale_pos_weight = 1.0
        except Exception:
            scale_pos_weight = 1.0
        clf = xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss', scale_pos_weight=scale_pos_weight)
        clf.fit(X_train, y_train)
    else:
        clf = xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss')
        # multiclass class-balanced via sample weights
        if getattr(args, 'class_balanced', False):
            try:
                classes = np.unique(y_train)
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                wmap = {int(c): float(w) for c, w in zip(classes, cw)}
                sample_weight = np.array([wmap[int(y)] for y in y_train], dtype=float)
            except Exception:
                sample_weight = None
        else:
            sample_weight = None
        if sample_weight is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


if HAS_TORCH:
    class SequenceDataset(Dataset):
        def __init__(self, X_seq, y):
            # X_seq: (N, T, D) or (N, D)
            self.X = torch.as_tensor(X_seq, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class CNN1D(nn.Module):
        def __init__(self, in_channels, n_classes, hidden_dim1=64, hidden_dim2=128, kernel_size=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim1, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(hidden_dim2, n_classes)

        def forward(self, x):
            # x shape: (B, T, D) -> transpose to (B, D, T)
            if x.dim() == 2:  # (B, D)
                x = x.unsqueeze(1)  # (B,1,D)
            x = x.transpose(1, 2)
            h = self.net(x).squeeze(-1)
            return self.fc(h)


    class RNNClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, n_classes=2, rnn_type='lstm', num_layers=2):
            super().__init__()
            if rnn_type.lower() == 'lstm':
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            else:
                self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, n_classes)
            print(f'RNNClassifier params: {self._count_params()}')
            print(self)

        def forward(self, x):
            # x: (B, T, D)
            out, _ = self.rnn(x)
            last = out[:, -1, :]
            return self.fc(last)
    
        def _count_params(module):
            return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


    def train_torch_model(model, train_loader, val_loader, args, device, epochs=None, lr=None, class_weights=None):
        """Train model on train_loader. Returns trained model.

        epochs and lr may be provided to override args for tuning.
        """
        model = model.to(device)
        _lr = lr if lr is not None else args.lr
        _epochs = epochs if epochs is not None else args.epochs
        opt = torch.optim.Adam(model.parameters(), lr=_lr)
        # class_weights: optional torch tensor on cpu; move to device for loss
        if class_weights is not None:
            cw = class_weights.to(device)
            print(f'Using class weights: {cw}')
            crit = nn.CrossEntropyLoss(weight=cw)
        else:
            crit = nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []

        # early stopping parameters
        patience = int(getattr(args, 'early_stopping_patience', 0))
        min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0))
        best_val = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        best_state = None

        for epoch in range(_epochs):
            model.train()
            running = 0.0
            n_samples = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                batch_n = xb.size(0)
                running += loss.item() * batch_n
                n_samples += batch_n
            avg_train_loss = running / n_samples if n_samples > 0 else 0.0
            train_losses.append(float(avg_train_loss))

            # compute validation loss if provided
            if val_loader is not None:
                model.eval()
                v_running = 0.0
                v_n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        loss = crit(logits, yb)
                        bv = xb.size(0)
                        v_running += loss.item() * bv
                        v_n += bv
                avg_val_loss = v_running / v_n if v_n > 0 else 0.0
                val_losses.append(float(avg_val_loss))

                # early stopping logic (only if patience > 0)
                if patience > 0:
                    # improvement if decrease in val loss by at least min_delta
                    if avg_val_loss + min_delta < best_val:
                        best_val = avg_val_loss
                        best_epoch = epoch
                        # store CPU copy of state dict for safe restore
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs). Restoring best epoch {best_epoch+1}.")
                            break
            else:
                val_losses.append(None)

            print(f"Epoch {epoch+1}/{_epochs}: Train Loss = {avg_train_loss:.4f}", end='')
            if val_loader is not None: print(f", Val Loss = {avg_val_loss:.4f}")

        history = {'train_loss': train_losses, 'val_loss': val_losses}

        # if we saved a best state, restore it onto the model (move tensors to device)
        if best_state is not None:
            mapped = {k: v.to(device) for k, v in best_state.items()}
            model.load_state_dict(mapped)

        return model, history


    def predict_on_loader(model, loader, device):
        model = model.to(device)
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                ys.append(yb.numpy())
                yps.append(preds)
        if ys:
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(yps)
        else:
            y_true = np.array([])
            y_pred = np.array([])
        return y_true, y_pred


def _need_normalize_for_model(model_name: str, args=None):
    # If user requested normalize-all, force normalization for all models
    if args is not None and getattr(args, 'normalize_all', False):
        return True
    # deep models need normalization; tree models typically not
    if model_name in ('cnn', 'lstm', 'gru'):
        return True
    return False


def _aggregate_features(X, agg='mean'):
    X = np.asarray(X)
    if X.ndim == 3:
        if agg == 'mean':
            return X.mean(axis=1)
        elif agg == 'max':
            return X.max(axis=1)
        elif agg == 'flatten':
            N, T, D = X.shape
            return X.reshape(N, T * D)
        else:
            return X.mean(axis=1)
    elif X.ndim == 2:
        return X
    else:
        raise ValueError('Unsupported X shape')


def tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, args):
    best = None
    best_score = -1.0
    best_params = None
    grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 10, 20, 30, 40, 50],    
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10, 20],    
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }
    trials = []
    for n in grid['n_estimators']:
        for md in grid['max_depth']:
            for mss in grid['min_samples_split']:
                for msl in grid['min_samples_leaf']:
                    params = {'n_estimators': int(n), 'max_depth': int(md) if md is not None else None, 'min_samples_split': int(mss), 'min_samples_leaf': int(msl)}
                    # choose classifier construction depending on qtype and balanced flag
                    if getattr(args, 'class_balanced', False):
                        clf = RandomForestClassifier(n_estimators=n, max_depth=md, random_state=args.seed, n_jobs=args.n_jobs, class_weight='balanced')
                    else:
                        clf = RandomForestClassifier(n_estimators=n, max_depth=md, random_state=args.seed, n_jobs=args.n_jobs)
                    clf.fit(X_train, y_train)
                    yp = clf.predict(X_val)
                    acc = accuracy_score(y_val, yp)
                    trials.append({'params': params, 'val_acc': float(acc)})
                    if acc > best_score:
                        best_score = acc
                        best = clf
                        best_params = params
    # retrain on train+val
    print('\tFinal refitting of RandomForestClassifier')
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    if getattr(args, 'class_balanced', False):
        print('Using class-balanced weights for RandomForestClassifier')
        final = RandomForestClassifier(**{k: best_params[k] for k in best_params}, random_state=args.seed, n_jobs=args.n_jobs, class_weight='balanced')
    else:
        final = RandomForestClassifier(**{k: best_params[k] for k in best_params}, random_state=args.seed, n_jobs=args.n_jobs)
    final.fit(X_comb, y_comb)
    y_pred = final.predict(X_test)
    # compute hyperparam importance from trials
    hyper_importance = compute_hyperparam_importance(trials)
    return final, y_pred, best_params, trials, hyper_importance


def tune_and_evaluate_xgb(X_train, y_train, X_val, y_val, X_test, y_test, args):
    if not HAS_XGB:
        raise ImportError('xgboost missing')
    best = None
    best_score = -1.0
    best_params = None
    grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 6, 8, 10],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001]
    }
    trials = []
    # grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 6],
    #     'learning_rate': [0.1, 0.01]
    # }
    for n in grid['n_estimators']:
        for md in grid['max_depth']:
            for lr in grid['learning_rate']:
                params = {'n_estimators': int(n), 'max_depth': int(md), 'learning_rate': float(lr)}
                # choose classifier construction depending on qtype and balanced flag
                if getattr(args, 'class_balanced', False) and getattr(args, 'qtype', 'binary') == 'binary':
                    # compute scale_pos_weight
                    try:
                        classes, counts = np.unique(y_train, return_counts=True)
                        if len(classes) == 2:
                            neg = int(counts[0])
                            pos = int(counts[1])
                            scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
                        else:
                            scale_pos_weight = 1.0
                    except Exception:
                        scale_pos_weight = 1.0
                    params['scale_pos_weight'] = float(scale_pos_weight)
                    clf = xgb.XGBClassifier(n_estimators=n, max_depth=md, learning_rate=lr, eval_metric='mlogloss', scale_pos_weight=scale_pos_weight)
                    clf.fit(X_train, y_train)
                else:
                    clf = xgb.XGBClassifier(n_estimators=n, max_depth=md, learning_rate=lr, eval_metric='mlogloss')
                    # multiclass or not class-balanced: use sample_weight only for multiclass balanced case
                    if getattr(args, 'class_balanced', False):
                        try:
                            classes = np.unique(y_train)
                            cw = compute_class_weight('balanced', classes=classes, y=y_train)
                            wmap = {int(c): float(w) for c, w in zip(classes, cw)}
                            sample_weight_train = np.array([wmap[int(y)] for y in y_train], dtype=float)
                        except Exception:
                            sample_weight_train = None
                    else:
                        sample_weight_train = None
                    if sample_weight_train is not None:
                        clf.fit(X_train, y_train, sample_weight=sample_weight_train)
                    else:
                        clf.fit(X_train, y_train)
                yp = clf.predict(X_val)
                acc = accuracy_score(y_val, yp)
                trials.append({'params': params, 'val_acc': float(acc)})
                if acc > best_score:
                    best_score = acc
                    best = clf
                    best_params = params
    # retrain on train+val
    print('\tFinal refitting of XGBoostClassifier')
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    # construct final classifier respecting qtype/class-balancing
    if getattr(args, 'class_balanced', False) and getattr(args, 'qtype', 'binary') == 'binary':
        # try:
        #     classes, counts = np.unique(y_comb, return_counts=True)
        #     if len(classes) == 2:
        #         neg = int(counts[0])
        #         pos = int(counts[1])
        #         scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
        #     else:
        #         scale_pos_weight = 1.0
        # except Exception:
        #     scale_pos_weight = 1.0
        final = xgb.XGBClassifier(**best_params, eval_metric='mlogloss')
        print('Using scale_pos_weight for XGBoost training')
        final.fit(X_comb, y_comb)
    else:
        final = xgb.XGBClassifier(**best_params, eval_metric='mlogloss')
        if getattr(args, 'class_balanced', False):
            try:
                classes = np.unique(y_comb)
                cw = compute_class_weight('balanced', classes=classes, y=y_comb)
                wmap = {int(c): float(w) for c, w in zip(classes, cw)}
                sample_weight_comb = np.array([wmap[int(y)] for y in y_comb], dtype=float)
                print('Using sample weights for XGBoost training')
            except Exception:
                sample_weight_comb = None
        else:
            sample_weight_comb = None
        if sample_weight_comb is not None:
            final.fit(X_comb, y_comb, sample_weight=sample_weight_comb)
        else:
            final.fit(X_comb, y_comb)
    # compute hyperparam importance
    hyper_importance = compute_hyperparam_importance(trials)
    y_pred = final.predict(X_test)
    return final, y_pred, best_params, trials, hyper_importance


def tune_and_evaluate_cnn(X_tr, y_tr, X_val, y_val, X_te, y_te, args, device):
    """Grid-search small CNN hyperparameter grid on train/val, retrain on train+val and test."""
    best_score = -1.0
    best_params = None
    best_model = None
    grid = {
        'lr': [1e-2, 1e-3, 1e-4],
        'epochs': [200],
        'hidden_dim1': [64, 128],
        'hidden_dim2': [128, 256],
        'kernel_size': [5, 9, 13]
    }
    # grid = {
    #     'lr': [1e-3, 1e-4],
    #     'epochs': [200],
    #     'hidden_dim1': [64, 128],
    #     'hidden_dim2': [128, 256],
    #     'kernel_size': [3, 5]
    # }
    trials = []
    for lr in grid['lr']:
        for epochs in grid['epochs']:
            for hidden_dim1 in grid['hidden_dim1']:
                for hidden_dim2 in grid['hidden_dim2']:
                    for kernel_size in grid['kernel_size']:
                        params = {'lr': lr, 'epochs': epochs, 'hidden_dim1': hidden_dim1, 
                                  'hidden_dim2': hidden_dim2, 'kernel_size': kernel_size}
                        print(params)
                        # build model
                        D = X_tr.shape[2]
                        model = CNN1D(in_channels=D, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, 
                                      kernel_size=kernel_size, n_classes=len(np.unique(y_tr)))
                        # dataloaders
                        tr_ds = SequenceDataset(X_tr, y_tr)
                        val_ds = SequenceDataset(X_val, y_val)
                        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
                        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                        # compute class weights from training labels and pass to loss
                        n_classes = int(np.max(y_tr)) + 1 if len(y_tr) > 0 else 1
                        if getattr(args, 'class_balanced', False):
                            try:
                                cw_arr = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_tr)
                            except Exception:
                                cw_arr = np.ones(n_classes, dtype=np.float32)
                        else:
                            cw_arr = np.ones(n_classes, dtype=np.float32)
                        cw = torch.as_tensor(cw_arr, dtype=torch.float32)
                        model, history = train_torch_model(model, tr_loader, val_loader, args, device, epochs=epochs, lr=lr, class_weights=cw)
                        y_true_val, y_pred_val = predict_on_loader(model, val_loader, device)
                        acc = accuracy_score(y_true_val, y_pred_val) if len(y_true_val) > 0 else 0.0
                        print(f'\tAcc_val={acc}')
                        trials.append({'params': params, 'val_acc': float(acc)})
                        if acc > best_score:
                            best_score = acc
                            best_params = params
                            best_model = model
                            best_history = history

    # retrain on train+val
    # X_comb = np.vstack([X_tr, X_val])
    # y_comb = np.concatenate([y_tr, y_val])
    # comb_ds = SequenceDataset(X_comb, y_comb)
    # comb_loader = DataLoader(comb_ds, batch_size=args.batch_size, shuffle=True)
    test_ds = SequenceDataset(X_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # final_model = CNN1D(in_channels=X_tr.shape[2], hidden_dim1=best_params['hidden_dim1'], 
    #                     hidden_dim2=best_params['hidden_dim2'], kernel_size=best_params['kernel_size'], n_classes=len(np.unique(y_comb)))
    # # compute class weights on combined train+val for final retrain
    # n_classes_comb = int(np.max(y_comb)) + 1 if len(y_comb) > 0 else 1
    # if getattr(args, 'class_balanced', False):
    #     try:
    #         cw_arr_comb = compute_class_weight('balanced', classes=np.arange(n_classes_comb), y=y_comb)
    #         print('Using balanced weights for CNN training')
    #     except Exception:
    #         cw_arr_comb = np.ones(n_classes_comb, dtype=np.float32)
    # else:
    #     cw_arr_comb = np.ones(n_classes_comb, dtype=np.float32)
    # cw_comb = torch.as_tensor(cw_arr_comb, dtype=torch.float32)
    # final_model, final_history = train_torch_model(final_model, comb_loader, None, args, device, epochs=best_params['epochs'], lr=best_params['lr'], class_weights=cw_comb)
    # y_true_test, y_pred_test = predict_on_loader(final_model, test_loader, device)

    y_true_test, y_pred_test = predict_on_loader(best_model, test_loader, device)
    hyper_importance = compute_hyperparam_importance(trials)
    return best_model, y_pred_test, best_params, best_history, trials, hyper_importance


def tune_and_evaluate_rnn(X_tr, y_tr, X_val, y_val, X_te, y_te, args, device, rnn_type='lstm'):
    best_score = -1.0
    best_params = None
    best_model = None
    grid = {
        'hidden_size': [64, 128, 256],
        'n_layers': [1, 2, 3],
        'lr': [1e-2, 1e-3, 1e-4],
        'epochs': [200]
    }
    # grid = {
    #     'hidden_size': [64, 128],
    #     'n_layers': [1, 2, 3],
    #     'lr': [1e-3, 1e-4],
    #     'epochs': [200]
    # }
    trials = []
    for hs in grid['hidden_size']:
        for nl in grid['n_layers']:
            for lr in grid['lr']:
                for epochs in grid['epochs']:
                    params = {'hidden_size': hs, 'lr': lr, 'epochs': epochs, 'n_layers': nl}
                    print(params)
                    # build model
                    model = RNNClassifier(input_size=X_tr.shape[2], hidden_size=hs, num_layers=nl, n_classes=len(np.unique(y_tr)), rnn_type=rnn_type)
                    tr_ds = SequenceDataset(X_tr, y_tr)
                    val_ds = SequenceDataset(X_val, y_val)
                    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                    # compute class weights for training labels
                    n_classes = int(np.max(y_tr)) + 1 if len(y_tr) > 0 else 1
                    if getattr(args, 'class_balanced', False):
                        try:
                            cw_arr = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_tr)
                        except Exception:
                            cw_arr = np.ones(n_classes, dtype=np.float32)
                    else:
                        cw_arr = np.ones(n_classes, dtype=np.float32)
                    cw = torch.as_tensor(cw_arr, dtype=torch.float32)
                    model, history = train_torch_model(model, tr_loader, val_loader, args, device, epochs=epochs, lr=lr, class_weights=cw)
                    y_true_val, y_pred_val = predict_on_loader(model, val_loader, device)
                    acc = accuracy_score(y_true_val, y_pred_val) if len(y_true_val) > 0 else 0.0
                    print(f'\tAcc_val={acc}')
                    trials.append({'params': params, 'val_acc': float(acc)})
                    if acc > best_score:
                        best_score = acc
                        best_params = params
                        best_model = model
                        best_history = history

    # retrain on train+val
    # X_comb = np.vstack([X_tr, X_val])
    # y_comb = np.concatenate([y_tr, y_val])
    # comb_ds = SequenceDataset(X_comb, y_comb)
    # comb_loader = DataLoader(comb_ds, batch_size=args.batch_size, shuffle=True)
    test_ds = SequenceDataset(X_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # final_model = RNNClassifier(input_size=X_tr.shape[2], hidden_size=best_params['hidden_size'], 
    #                             num_layers=best_params['n_layers'], n_classes=len(np.unique(y_comb)), rnn_type=rnn_type)
    # # compute class weights on combined train+val
    # n_classes_comb = int(np.max(y_comb)) + 1 if len(y_comb) > 0 else 1
    # if getattr(args, 'class_balanced', False):
    #     try:
    #         cw_arr_comb = compute_class_weight('balanced', classes=np.arange(n_classes_comb), y=y_comb)
    #         print('Using balanced weights for RNN training')
    #     except Exception:
    #         cw_arr_comb = np.ones(n_classes_comb, dtype=np.float32)
    # else:
    #     cw_arr_comb = np.ones(n_classes_comb, dtype=np.float32)
    # cw_comb = torch.as_tensor(cw_arr_comb, dtype=torch.float32)
    # final_model, final_history = train_torch_model(final_model, comb_loader, None, args, device, 
    #                                                epochs=best_params['epochs'], lr=best_params['lr'], class_weights=cw_comb)
    # y_true_test, y_pred_test = predict_on_loader(final_model, test_loader, device)

    y_true_test, y_pred_test = predict_on_loader(best_model, test_loader, device)
    hyper_importance = compute_hyperparam_importance(trials)
    return best_model, y_pred_test, best_params, best_history, trials, hyper_importance


def run_models(args):
    X, y, feat_id = load_npz(args.npz, label_key=args.label_key)
    if y is None:
        raise ValueError('No labels found; please supply --label-key or include labels in the npz')

    if args.class_balanced:
        outdir = args.out_dir + '_balanced'
    else:
        outdir = args.out_dir + '_unbalanced'
    outdir = Path(outdir)
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # parse ratios
    if hasattr(args, 'ratios') and args.ratios:
        try:
            ratios = tuple(int(x) for x in args.ratios.split(','))
            if len(ratios) != 3:
                ratios = (6, 1, 3)
        except Exception:
            ratios = (6, 1, 3)
    else:
        ratios = (6, 1, 3)

    groups = split_person_groups(feat_ids=feat_id, seed=args.seed, ratios=ratios)
    train_idx = groups['train_idx']
    val_idx = groups['val_idx']
    test_idx = groups['test_idx']

    # encode labels across full set
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels = list(map(str, le.classes_))

    # aggregate features for static/tree models
    X_static = _aggregate_features(X, agg=args.agg)

    results = {'persons': {'train': groups['train_persons'], 'val': groups['val_persons'], 'test': groups['test_persons']}}

    def get_split_stats(y_all, idxs, labels):
        """Return counts and percentages per class for the given indices."""
        stats = {}
        if idxs is None or len(idxs) == 0:
            for lab in labels:
                stats[lab] = {'count': 0, 'pct': 0.0}
            return {'n_samples': 0, 'per_class': stats}
        y = np.asarray(y_all)[idxs]
        n = len(y)
        counts = np.bincount(y, minlength=len(labels))
        for i, lab in enumerate(labels):
            cnt = int(counts[i])
            pct = float(cnt) / n if n > 0 else 0.0
            stats[lab] = {'count': cnt, 'pct': round(pct, 4)}
        return {'n_samples': int(n), 'per_class': stats}


    # Helper to scale given model needs
    def scale_if_needed(model_name, Xtr, Xv, Xt):
        need = _need_normalize_for_model(model_name, args)
        if need:
            scaler = StandardScaler()
            scaler.fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xv_s = scaler.transform(Xv) if Xv is not None and len(Xv) > 0 else Xv
            Xt_s = scaler.transform(Xt) if Xt is not None and len(Xt) > 0 else Xt
            return Xtr_s, Xv_s, Xt_s, scaler
        else:
            return Xtr, Xv, Xt, None

    # compute and record split statistics
    split_stats = {
        'train': get_split_stats(y_enc, train_idx, labels),
        'val': get_split_stats(y_enc, val_idx, labels),
        'test': get_split_stats(y_enc, test_idx, labels),
    }
    results['split_stats'] = split_stats

    # Print split statistics
    print('\nSplit statistics:')
    for k in ('train', 'val', 'test'):
        s = split_stats[k]
        print(f"{k}: {s['n_samples']} samples")
        for lab, info in s['per_class'].items():
            print(f"  {lab}: {info['count']} ({info['pct']*100:.2f}%)")

    # Random Forest
    def _run_tree_model(name, estimator_factory, tune_fn=None):
        """Helper to run a tree-based model with optional tuning.
        estimator_factory: callable(args) -> estimator instance (unfitted)
        tune_fn: optional function(X_tr, y_tr, X_val, y_val, X_te, y_te, args) -> (final_model, y_pred, best_params)
        """
        X_tr = X_static[train_idx]
        X_val = X_static[val_idx]
        X_te = X_static[test_idx]
        y_tr = y_enc[train_idx]
        y_val = y_enc[val_idx]
        y_te = y_enc[test_idx]

        X_tr_s, X_val_s, X_te_s, _ = scale_if_needed(name, X_tr, X_val, X_te)

        if args.tune and tune_fn is not None:
            _, y_pred, best_params, trials, hyper_importance = tune_fn(X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te, args)
            rep = metrics_report(y_te, y_pred, labels=range(len(labels)))
            rep = {'best_params': best_params, 'trials': trials, 'hyper_importance': hyper_importance, **rep}
        else:
            # special handling for XGBoost when class-balanced option is enabled
            if name == 'XGBoost' and getattr(args, 'class_balanced', False) and HAS_XGB:
                if getattr(args, 'qtype', 'binary') == 'binary':
                    # set scale_pos_weight based on train counts
                    try:
                        classes, counts = np.unique(y_tr, return_counts=True)
                        if len(classes) == 2:
                            neg = int(counts[0])
                            pos = int(counts[1])
                            scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
                        else:
                            scale_pos_weight = 1.0
                    except Exception:
                        scale_pos_weight = 1.0
                    clf = xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss', scale_pos_weight=scale_pos_weight)
                    clf.fit(X_tr_s, y_tr)
                else:
                    # multiclass: compute per-sample weights and pass to fit
                    clf = xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss')
                    try:
                        classes = np.unique(y_tr)
                        cw = compute_class_weight('balanced', classes=classes, y=y_tr)
                        wmap = {int(c): float(w) for c, w in zip(classes, cw)}
                        sample_weight_train = np.array([wmap[int(y)] for y in y_tr], dtype=float)
                    except Exception:
                        sample_weight_train = None
                    if sample_weight_train is not None:
                        clf.fit(X_tr_s, y_tr, sample_weight=sample_weight_train)
                    else:
                        clf.fit(X_tr_s, y_tr)
            else:
                clf = estimator_factory(args)
                clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            rep = metrics_report(y_te, y_pred, labels=range(len(labels)))

        results[name] = rep
        plot_confusion(rep['confusion'], labels, str(outdir / f'confusion_{name}'), model_name=name, accuracy=rep.get('accuracy'))
        plot_per_class_metrics(rep['per_class'], labels, str(outdir / f'perclass_{name}'), model_name=name, accuracy=rep.get('accuracy'))

    # run RF and XGB via helper
    if 'rf' in args.models:
        _run_tree_model('RandomForest', lambda a: RandomForestClassifier(n_estimators=200, random_state=a.seed, n_jobs=a.n_jobs), tune_and_evaluate_rf if args.tune else None)

    if 'xgb' in args.models:
        if not HAS_XGB:
            print('xgboost not installed; skipping xgb')
        else:
            _run_tree_model('XGBoost', lambda a: xgb.XGBClassifier(n_estimators=200, eval_metric='mlogloss'), tune_and_evaluate_xgb if args.tune else None)

    # Torch models: build sequence splits based on person
    if HAS_TORCH and any(m in args.models for m in ('cnn', 'lstm', 'gru')):
        seed_everything(args.seed)
        
        Xnp = np.asarray(X)
        if Xnp.ndim == 2:
            X_seq = Xnp[:, None, :]
        else:
            X_seq = Xnp

        X_train_seq = X_seq[train_idx]
        X_val_seq = X_seq[val_idx]
        X_test_seq = X_seq[test_idx]
        y_train_seq = y_enc[train_idx]
        y_val_seq = y_enc[val_idx]
        y_test_seq = y_enc[test_idx]

        # scale per-model if needed (fit on train only across timesteps)
        def scale_seq_if_needed(model_name, Xtr, Xv, Xt):
            need = _need_normalize_for_model(model_name, args)
            if need:
                N, T, D = Xtr.shape
                flat = Xtr.reshape(N * T, D)
                scaler = StandardScaler()
                scaler.fit(flat)
                Xtr_s = scaler.transform(flat).reshape(N, T, D)
                Xv_s = None
                Xt_s = None
                if Xv is not None and len(Xv) > 0:
                    Nv = Xv.shape[0]
                    flatv = Xv.reshape(Nv * Xv.shape[1], Xv.shape[2])
                    Xv_s = scaler.transform(flatv).reshape(Nv, Xv.shape[1], Xv.shape[2])
                if Xt is not None and len(Xt) > 0:
                    Nt = Xt.shape[0]
                    flatt = Xt.reshape(Nt * Xt.shape[1], Xt.shape[2])
                    Xt_s = scaler.transform(flatt).reshape(Nt, Xt.shape[1], Xt.shape[2])
                return Xtr_s, Xv_s, Xt_s, scaler
            else:
                return Xtr, Xv, Xt, None

        # prepare DataLoaders
        def _run_sequence_model(name, model_builder, tune_fn=None, rnn_type=None):
            """Helper to run a sequence model (CNN or RNN) with optional tuning."""
            # Xtr_s, Xval_s, Xte_s, train_loader, val_loader, test_loader, y_* are in outer scope
            if args.tune and tune_fn is not None:
                    res = tune_fn(Xtr_s, y_train_seq, Xval_s, y_val_seq, Xte_s, y_test_seq, args, device) if tune_fn is not None else (None, None, None)
                    # tune_fn may return one of:
                    # (model, y_pred, best_params)
                    # (model, y_pred, best_params, history)
                    # (model, y_pred, best_params, history, trials, hyper_importance)
                    final_model = None
                    y_pred = np.array([])
                    best_params = None
                    history = None
                    trials = None
                    hyper_importance = None
                    if isinstance(res, tuple):
                        if len(res) == 6:
                            final_model, y_pred, best_params, history, trials, hyper_importance = res
                        elif len(res) == 4:
                            final_model, y_pred, best_params, history = res
                        elif len(res) == 3:
                            final_model, y_pred, best_params = res
                        else:
                            # best-effort: take first three
                            try:
                                final_model, y_pred, best_params = res[:3]
                            except Exception:
                                final_model, y_pred, best_params = (None, np.array([]), None)
                    rep = metrics_report(y_test_seq, y_pred, labels=range(len(labels)))
                    results[name] = {'best_params': best_params, 'trials': trials, 'hyper_importance': hyper_importance, **rep}
                    # plot loss if available
                    try:
                        plot_loss(history, str(outdir / f'loss_{name}'), model_name=name)
                    except Exception:
                        pass
            else:
                model = model_builder(Xtr_s.shape[2], len(labels), args)
                # compute class weights from train labels and pass to training
                n_cls = int(np.max(y_train_seq)) + 1 if len(y_train_seq) > 0 else 1
                if getattr(args, 'class_balanced', False):
                    try:
                        cw_arr_local = compute_class_weight('balanced', classes=np.arange(n_cls), y=y_train_seq)
                    except Exception:
                        cw_arr_local = np.ones(n_cls, dtype=np.float32)
                else:
                    cw_arr_local = np.ones(n_cls, dtype=np.float32)
                cw_local = torch.as_tensor(cw_arr_local, dtype=torch.float32)
                model, history = train_torch_model(model, train_loader, val_loader, args, device, class_weights=cw_local)
                y_true, y_pred = predict_on_loader(model, test_loader, device)
                rep = metrics_report(y_true, y_pred, labels=range(len(labels)))
                results[name] = rep
                # plot loss curve
                try:
                    plot_loss(history, str(outdir / f'loss_{name}'), model_name=name)
                except Exception:
                    pass

            plot_confusion(results[name]['confusion'], labels, str(outdir / f'confusion_{name}'), 
                           model_name=name, accuracy=results[name].get('accuracy'))
            plot_per_class_metrics(results[name]['per_class'], labels, str(outdir / f'perclass_{name}'), 
                                   model_name=name, accuracy=results[name].get('accuracy'))

        for model_name in ('cnn', 'lstm', 'gru'):
            if model_name not in args.models:
                continue
            Xtr_s, Xval_s, Xte_s, _ = scale_seq_if_needed(model_name, X_train_seq, X_val_seq, X_test_seq)

            train_ds = SequenceDataset(Xtr_s, y_train_seq)
            val_ds = SequenceDataset(Xval_s, y_val_seq) if Xval_s is not None and len(Xval_s) > 0 else None
            test_ds = SequenceDataset(Xte_s, y_test_seq)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else test_loader

            device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

            # map to model builders and tune functions
            if model_name == 'cnn':
                _run_sequence_model('CNN', lambda D, nc, a: CNN1D(
                    in_channels=D, hidden_dim1=a.hidden_dim1, hidden_dim2=a.hidden_dim2, 
                    kernel_size=a.kernel_size, n_classes=nc), tune_and_evaluate_cnn)

            if model_name in ('lstm', 'gru'):
                rnn_type = 'lstm' if model_name == 'lstm' else 'gru'
                _run_sequence_model(rnn_type.upper(), lambda D, nc, a: RNNClassifier(
                    input_size=D, hidden_size=a.hidden_size, n_classes=nc, rnn_type=rnn_type), 
                    tune_and_evaluate_rnn, rnn_type=rnn_type)

    # print which persons are in each set
    print('Persons in train:', groups['train_persons'])
    print('Persons in val:  ', groups['val_persons'])
    print('Persons in test: ', groups['test_persons'])

    # build a final comparison table
    comparison = []
    for model_name, rep in results.items():
        if model_name == 'persons' or model_name == 'split_stats':
            continue
        if not isinstance(rep, dict):
            continue
        acc = rep.get('accuracy')
        macro_f1 = rep.get('macro_f1')
        best_params = rep.get('best_params') if 'best_params' in rep else None
        comparison.append({'model': model_name, 'accuracy': acc, 'macro_f1': macro_f1, 'best_params': best_params})

    results['comparison'] = comparison

    # print comparison table
    if comparison:
        print('\nModel comparison:')
        # compute column widths
        cols = ['model', 'accuracy', 'macro_f1', 'best_params']
        rows = [cols]
        for c in comparison:
            rows.append([str(c.get(k, '')) for k in cols])
        col_widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
        # header
        hdr = ' | '.join(r.ljust(col_widths[i]) for i, r in enumerate(rows[0]))
        sep = '-+-'.join('-' * col_widths[i] for i in range(len(cols)))
        print(hdr)
        print(sep)
        for r in rows[1:]: 
            print(' | '.join(r[i].ljust(col_widths[i]) for i in range(len(cols))))

    # save JSON report including persons per split
    res_path = outdir / 'results.json'
    with open(res_path, 'w') as f:
        json.dump({'models': results, 'labels': labels, 'persons': results.get('persons', {})}, f, indent=2)

    print('\nDone. Results written to', res_path)
    # TODO: add more data and then align VLP and ML results?


def parse_args():
    p = argparse.ArgumentParser()
    qtype = 'binary_trajvelpod'
    p.add_argument('--npz', default=f'features/ml/{qtype}.npz', help='path to features npz file')
    p.add_argument('--label-key', dest='label_key', default=1, help='key in npz containing labels (index into feat_id or a key name)')
    p.add_argument('--out-dir', dest='out_dir', default=f'logs/ml/{qtype}', help='output directory')
    p.add_argument('--models', nargs='+', default=['lstm'], help='models to run: rf xgb cnn lstm gru')
    # p.add_argument('--models', nargs='+', default=['rf', 'xgb', 'cnn', 'lstm'], help='models to run: rf xgb cnn lstm gru')
    p.add_argument('--agg', choices=['mean', 'max', 'flatten'], default='mean', help='aggregation method for tree models')
    p.add_argument('--test-size', type=float, default=0.3)
    p.add_argument('--ratios', type=str, default='6,1,3', help='train,val,test ratio as comma separated ints (default 6,1,3)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--normalize', action='store_true', default=False, help='(deprecated) apply StandardScaler to all models; prefer --normalize-all or model-adaptive default')
    p.add_argument('--normalize-all', action='store_true', default=False, help='apply StandardScaler for all models')
    p.add_argument('--tune', action='store_true', default=False, help='perform hyperparameter tuning on validation set for RF and XGBoost')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--early-stopping-patience', type=int, default=30, help='patience (in epochs) for early stopping of deep models; 0 to disable')
    p.add_argument('--early-stopping-min-delta', type=float, default=0.0, help='minimum change in val loss to qualify as improvement for early stopping')
    p.add_argument('--class-balanced', dest='class_balanced', action='store_true', default=True, help='use class-balanced weighting for deep-model losses (CrossEntropy)')
    p.add_argument('--qtype', choices=['binary', 'multi'], default='binary', help='question type: binary or multi (multiclass). Affects XGBoost weighting behavior')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--use-cuda', action='store_true', default=True)
    p.add_argument('--overwrite', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    # adapt param names from npz arg
    if not os.path.exists(args.npz):
        raise FileNotFoundError(args.npz)
    run_models(args)


if __name__ == '__main__':
    main()