"""Load the authors' basic anchor-free DSNet weights."""

from pathlib import Path

import torch
import yaml

from .reference.dsnet_af import DSNetAF


REVISION = "1804176e2e8b57846beb063667448982273fca89"
FOLD = 3
CHECKPOINT_NAME = f"tvsum.yml.{FOLD}.pt"
FEATURE_DIM = 1024
HIDDEN_DIM = 128
ATTENTION_HEADS = 8


def split_role(feature_key):
    splits = Path(__file__).with_name("reference") / "tvsum.yml"
    split = yaml.safe_load(splits.read_text(encoding="utf-8"))[FOLD]
    matches = [role for role in ("test", "train")
               if any(Path(key).name == feature_key for key in split[f"{role}_keys"])]
    if len(matches) != 1:
        raise ValueError(f"Feature group {feature_key} is not uniquely assigned in fold {FOLD}")
    return matches[0]


def load_model(checkpoint):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing fold {FOLD} anchor-free DSNet checkpoint at {checkpoint}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = DSNetAF(FEATURE_DIM, HIDDEN_DIM, ATTENTION_HEADS)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.eval()
