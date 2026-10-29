"""Compatibility loader for cached specialist test predictions.

This module does not load trained model weights or run specialist inference.
Arrays preserve the original competition test row order.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT.parent

class _TorchStub:
    @staticmethod
    def set_num_threads(n): pass
    @staticmethod
    def set_num_interop_threads(n): pass
    @staticmethod
    def set_flush_denormal(flag): pass

try:
    import torch
except ImportError:
    torch = _TorchStub()

class _OldSources:
    @staticmethod
    def sources():
        tr_p = WORK_DIR / "train.csv"
        te_p = WORK_DIR / "test.csv"
        sa_p = WORK_DIR / "sample_submission.csv"
        train_raw = pd.read_csv(tr_p) if tr_p.exists() else pd.DataFrame()
        test_raw = pd.read_csv(te_p) if te_p.exists() else pd.DataFrame()
        sample_raw = pd.read_csv(sa_p) if sa_p.exists() else pd.DataFrame()
        return train_raw, test_raw, sample_raw

old = _OldSources()

def load_preparations():
    npz_path = ROOT / "specialist_test_predictions.npz"
    return dict(np.load(npz_path))

def predict(preps, test_raw=None):
    comp_test = {k: preps[k] for k in ("p1", "p2", "p3", "p5", "p6", "p8")}
    mu_miss_aligned = preps["mu_miss_aligned"]
    return mu_miss_aligned, comp_test, {}
