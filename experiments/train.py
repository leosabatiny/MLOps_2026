import argparse
from pathlib import Path

import torch
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer

# Support both naming styles depending on how your utils were implemented/exported.
try:
    from ml_core.utils import load_config, seed_everything, setup_logger
except ImportError:
    from ml_core.utils import loadconfig as load_config
    from ml_core.utils import seedeverything as seed_everything
    from ml_core.utils import setuplogger as setup_logger


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 5e-4))
    weight_decay = float(opt_cfg.get("weightdecay", 0.0))

    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer name: {name}")


def main(args):
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    Path(config["training"]["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = MLP(
        inputshape=tuple(config["model"]["inputshape"]),
        hiddenunits=list(config["model"]["hiddenunits"]),
        dropoutrate=float(config["model"]["dropoutrate"]),
        numclasses=int(config["model"]["numclasses"]),
    )

    optimizer = build_optimizer(model, config)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)

