"""
model_summary.py
================
Print torchinfo summaries for all model variants in this project.

Usage
-----
    python model_summary.py              # all models, batch_size=2
    python model_summary.py --model net  # only Net  (ResNet1D + ResNet2D)
    python model_summary.py --model net2 # only Net2 (ResNet1D + AST)
    python model_summary.py --bs 8       # custom batch size

Requires
--------
    pip install torchinfo
"""

import argparse
import torch
from torchinfo import summary

from net import Net, Net2
from resnet.resnet_1D import CreateResNet1D
from resnet.resnet_2D import CreateResNet2D
from resnet.ast_2D import CreateAST2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Input shapes ──────────────────────────────────────────────────────────────
SPECT_SHAPE = (3, 128, 126)   # [C, freq_bins, time_frames]
AUDIO_SHAPE = (1, 16000)      # [C, samples]  — 1 s @ 16 kHz
NUM_CLASSES = 10


def sep(title=""):
    width = 80
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def summarize_encoders(batch_size):
    """Print summaries for each encoder sub-module individually."""
    sep("ResNet18-1D  (raw-audio encoder)   input: [B, 1, 16000]")
    summary(
        CreateResNet1D(num_classes=NUM_CLASSES).to(device),
        input_size=(batch_size, *AUDIO_SHAPE),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )

    sep("ResNet18-2D  (spectrogram encoder)  input: [B, 3, 128, 126]")
    summary(
        CreateResNet2D(img_channels=3, num_classes=NUM_CLASSES).to(device),
        input_size=(batch_size, *SPECT_SHAPE),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )

    sep("AudioSpecTransformer (AST-lite)     input: [B, 3, 128, 126]")
    summary(
        CreateAST2D(img_channels=3, num_classes=NUM_CLASSES).to(device),
        input_size=(batch_size, *SPECT_SHAPE),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )


def summarize_net(batch_size, unsupervised=False):
    mode = "unsupervised" if unsupervised else "supervised"
    sep(f"Net  (ResNet1D + ResNet2D)  —  {mode}  input: spect + audio")
    model = Net(img_channels=3, num_classes=NUM_CLASSES,
                unsupervised=unsupervised).to(device)
    summary(
        model,
        input_size=[(batch_size, *SPECT_SHAPE), (batch_size, *AUDIO_SHAPE)],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )


def summarize_net2(batch_size, unsupervised=False):
    mode = "unsupervised" if unsupervised else "supervised"
    sep(f"Net2 (ResNet1D + AST-lite)  —  {mode}  input: spect + audio")
    model = Net2(img_channels=3, num_classes=NUM_CLASSES,
                 unsupervised=unsupervised).to(device)
    summary(
        model,
        input_size=[(batch_size, *SPECT_SHAPE), (batch_size, *AUDIO_SHAPE)],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
    )


def main():
    parser = argparse.ArgumentParser(description="Print torchinfo model summaries")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "encoders", "net", "net2"],
                        help="Which model(s) to summarise (default: all)")
    parser.add_argument("--bs", type=int, default=2,
                        help="Batch size for the summary (default: 2)")
    args = parser.parse_args()

    print(f"\nDevice : {device}")
    print(f"Batch  : {args.bs}")

    if args.model in ("all", "encoders"):
        summarize_encoders(args.bs)

    if args.model in ("all", "net"):
        summarize_net(args.bs, unsupervised=False)
        summarize_net(args.bs, unsupervised=True)

    if args.model in ("all", "net2"):
        summarize_net2(args.bs, unsupervised=False)
        summarize_net2(args.bs, unsupervised=True)

    sep()


if __name__ == "__main__":
    main()
