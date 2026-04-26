import argparse
from dataclasses import replace
from pathlib import Path

import torch

from dcic_ensemble_pipeline import (
    DEFAULT_ENCODERS,
    EntmaxImageExperimentConfig,
    build_run_name,
    config_to_dict,
    list_image_datasets,
    run_dataset_experiment,
    save_results_summary,
    write_json,
)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets") # if not set, all datasets found in `data_root` are used, can be passed multiple times
    parser.add_argument("--encoder", default="resnet18", choices=sorted(DEFAULT_ENCODERS)) # which torchvision encoder to use (e.g. also resnet50, check DEFAULT_ENCODERS in main pipeline for all options)
    parser.add_argument("--ensemble-size", type=int, default=25) # number of independently initialized models trained on the same split
    parser.add_argument("--epochs", type=int, default=20) # number of training epochs
    parser.add_argument("--batch-size", type=int, default=32) # batch size used for train, validation and test dataloaders
    parser.add_argument("--lr", type=float, default=1e-3) # learning rate for AdamW
    parser.add_argument("--weight-decay", type=float, default=1e-4) # weight decay for AdamW
    parser.add_argument("--validation-size", type=float, default=0.1) # fraction of the non-test data used as validation set
    parser.add_argument("--patience", type=int, default=4)  # number of epochs with no improvement on val set after which training is stopped
    parser.add_argument("--seed", type=int, default=42) # random seed used for splitting and model initialization
    parser.add_argument("--workers", type=int, default=4) # number of dataloader worker processes
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--finetune", action="store_true")  # if not set, encoder is frozen and only classification head is trained
    parser.add_argument("--no-pretrained", action="store_true")  # if set, encoder is initialized with random weights instead of imagenet pretrained weights
    parser.add_argument("--test-fold", default="fold1")  # which fold to use as test set, options are: fold1, fold2, fold3, fold4, fold5
    parser.add_argument("--output-root", type=Path, default=Path("out/image"))
    parser.add_argument("--data-root", type=Path, default=Path("data/image"))
    parser.add_argument("--augmentation", choices=["none", "basic"], default="basic")  # basic is currently just RandomHorizontalFlip
    parser.add_argument("--dropout", type=float, default=0.0)  # dropout rate for classification head, applied after global average pooling and before linear layer
    parser.add_argument("--entmax-alpha", type=float, default=1.0)  # alpha parameter for entmax, controls sparsity output logits, setting alpha=1 will use softmax + CE, alpha=2 is sparsemax
    return parser


def main():
    args = build_parser().parse_args()
    datasets = args.datasets or list_image_datasets(args.data_root)

    config = EntmaxImageExperimentConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        encoder_name=args.encoder,
        pretrained=not args.no_pretrained,
        freeze_encoder=not args.finetune,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        early_stopping_patience=args.patience,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
        test_fold=args.test_fold,
        augmentation=args.augmentation,
        classifier_dropout=args.dropout,
        entmax_alpha=args.entmax_alpha,
    )

    run_root = config.output_root / build_run_name(config)
    run_root.mkdir(parents=True, exist_ok=True)
    config = replace(config, output_root=run_root)
    write_json(run_root / "config.json", {"datasets": datasets, "config": config_to_dict(config)})

    print(f"Run directory: {run_root}")

    results = []
    for dataset_name in datasets:
        print(f"\nRunning entmax on {dataset_name}")
        result = run_dataset_experiment(dataset_name, config)
        print(f"Ensemble cross entropy: {result.mean_ensemble_cross_entropy:.4f}")
        results.append(result)
        save_results_summary(run_root, results)

    print("\nEnsemble cross entropy results:")
    for result in results:
        print(f"{result.dataset_name}: ensemble CE={result.mean_ensemble_cross_entropy:.4f}")


if __name__ == "__main__":
    main()
