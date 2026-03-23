import argparse as ap

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from ._cli import (
    add_arguments_emb,
    add_arguments_tx,
    run_emb_fit,
    run_emb_transform,
    run_emb_query,
    run_emb_preprocess,
    run_emb_eval,
    run_tx_infer,
    run_tx_evaluate,
    run_tx_preprocess_train,
    run_tx_train,
)


def get_args() -> tuple[ap.Namespace, list[str]]:
    """Parse known args and return remaining args for Hydra overrides"""
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_emb(subparsers.add_parser("emb"))
    add_arguments_tx(subparsers.add_parser("tx"))

    # Use parse_known_args to get both known args and remaining args
    return parser.parse_args()


def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra config with optional overrides.

    Supports scale= shorthand for SE model:
        python -m src.state emb fit scale=2b experiment.name=my_run
    This loads configs/scale/2b.yaml and merges its model overrides
    before applying remaining CLI overrides.
    """
    if overrides is None:
        overrides = []

    # Extract scale= override if present (not a Hydra config group, just a convenience)
    scale_name = None
    remaining_overrides = []
    for o in overrides:
        if o.startswith("scale="):
            scale_name = o.split("=", 1)[1]
        else:
            remaining_overrides.append(o)

    # Initialize Hydra with the path to your configs directory
    # Adjust the path based on where this file is relative to configs/
    with initialize(version_base=None, config_path="configs"):
        match method:
            case "emb":
                cfg = compose(config_name="state-defaults", overrides=remaining_overrides)
            case "tx":
                cfg = compose(config_name="config", overrides=remaining_overrides)
            case _:
                raise ValueError(f"Unknown method: {method}")

    # Merge scale config if specified
    if scale_name is not None:
        from pathlib import Path
        scale_path = Path(__file__).parent / "configs" / "scale" / f"{scale_name}.yaml"
        if not scale_path.exists():
            available = [f.stem for f in scale_path.parent.glob("*.yaml")]
            raise ValueError(f"Unknown scale '{scale_name}'. Available: {available}")
        scale_cfg = OmegaConf.load(scale_path)
        cfg = OmegaConf.merge(cfg, scale_cfg)
        # Re-apply CLI overrides so they take precedence over the scale config
        if remaining_overrides:
            cli_overrides = OmegaConf.from_dotlist(remaining_overrides)
            cfg = OmegaConf.merge(cfg, cli_overrides)

    return cfg


def show_hydra_help(method: str):
    """Show Hydra configuration help with all parameters"""
    from omegaconf import OmegaConf

    # Load the default config to show structure
    cfg = load_hydra_config(method)

    print("Hydra Configuration Help")
    print("=" * 50)
    print(f"Configuration for method: {method}")
    print()
    print("Full configuration structure:")
    print(OmegaConf.to_yaml(cfg))
    print()
    print("Usage examples:")
    print("  Override single parameter:")
    print("    uv run state tx train data.batch_size=64")
    print()
    print("  Override nested parameter:")
    print("    uv run state tx train model.kwargs.hidden_dim=512")
    print()
    print("  Override multiple parameters:")
    print("    uv run state tx train data.batch_size=64 training.lr=0.001")
    print()
    print("  Change config group:")
    print("    uv run state tx train data=custom_data model=custom_model")
    print()
    print("Available config groups:")

    # Show available config groups
    from pathlib import Path

    config_dir = Path(__file__).parent / "configs"
    if config_dir.exists():
        for item in config_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                configs = [f.stem for f in item.glob("*.yaml")]
                if configs:
                    print(f"  {item.name}: {', '.join(configs)}")

    exit(0)


def main():
    args = get_args()

    match args.command:
        case "emb":
            match args.subcommand:
                case "fit":
                    cfg = load_hydra_config("emb", args.hydra_overrides)
                    run_emb_fit(cfg, args)
                case "transform":
                    run_emb_transform(args)
                case "query":
                    run_emb_query(args)
                case "preprocess":
                    run_emb_preprocess(args)
                case "eval":
                    run_emb_eval(args)
        case "tx":
            match args.subcommand:
                case "train":
                    if hasattr(args, "help") and args.help:
                        # Show Hydra configuration help
                        show_hydra_help("tx")
                    else:
                        # Load Hydra config with overrides for sets training
                        cfg = load_hydra_config("tx", args.hydra_overrides)
                        run_tx_train(cfg)
                case "evaluate":
                    run_tx_evaluate(args)
                case "infer":
                    # Run inference using argparse, similar to predict
                    run_tx_infer(args)
                case "preprocess_train":
                    # Run preprocessing using argparse
                    run_tx_preprocess_train(args)


if __name__ == "__main__":
    main()
