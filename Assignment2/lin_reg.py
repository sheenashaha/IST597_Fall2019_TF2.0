#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 599 - Foundations of Deep Learning
Assignment #00001 - Problem 1: Linear Regression (TF2, NO Keras)
Author: Sheena Shaha
"""

import argparse
import os
import time
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------

def name_to_seed(name: str) -> int:
    """Convert a first name to a numeric seed (ASCII -> decimal)."""
    if not name:
        return 1337
    return int.from_bytes(name.strip().encode("utf-8"), "little") % (2**31 - 1)


def set_seed_all(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_noise_arg(arg):
    """Return a noise function based on the command-line argument."""
    if arg is None or arg.lower() == "none":
        return lambda n: np.zeros((n,), dtype=np.float32)
    if arg.startswith("std="):
        std = float(arg.split("=")[1])
        return lambda n: np.random.normal(0, std, size=(n,)).astype(np.float32)
    if arg.startswith("uniform="):
        a, b = map(float, arg.split("=")[1].split(","))
        return lambda n: np.random.uniform(a, b, size=(n,)).astype(np.float32)
    raise ValueError(f"Invalid noise spec: {arg}")


def make_loss_fn(spec):
    """Create the loss function based on user input."""
    s = spec.lower()
    if s == "l2":
        return lambda y, yhat: tf.reduce_mean(tf.square(y - yhat)), "L2 (MSE)"
    if s == "l1":
        return lambda y, yhat: tf.reduce_mean(tf.abs(y - yhat)), "L1 (MAE)"
    if s.startswith("huber"):
        delta = 1.0
        if ":" in s:
            delta = float(s.split("delta=")[1])
        def huber(y, yhat):
            r = tf.abs(y - yhat)
            quad = tf.minimum(r, delta)
            lin = r - quad
            return tf.reduce_mean(0.5 * tf.square(quad) + delta * lin)
        return huber, f"Huber (δ={delta})"
    if s.startswith("hybrid"):
        alpha = 0.5
        if ":" in s:
            alpha = float(s.split("alpha=")[1])
        def hybrid(y, yhat):
            l1 = tf.reduce_mean(tf.abs(y - yhat))
            l2 = tf.reduce_mean(tf.square(y - yhat))
            return alpha * l1 + (1 - alpha) * l2
        return hybrid, f"Hybrid (α={alpha})"
    raise ValueError(f"Unsupported loss: {spec}")


# -------------------------------------------------------------
# Data Generation
# -------------------------------------------------------------
def generate_data(n=10000, noise_fn=None):
    """Generate synthetic (x, y) pairs following f(x) = 3x + 2 + noise."""
    x = np.random.uniform(-5, 5, size=(n,)).astype(np.float32)
    y_clean = 3 * x + 2
    noise = noise_fn(n) if noise_fn else np.zeros_like(x)
    y = y_clean + noise
    return x, y, y_clean


# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------
def train(args):
    outdir = "outputs_linear"
    os.makedirs(outdir, exist_ok=True)

    seed = name_to_seed(args.seed_name) if args.seed_name else args.seed
    set_seed_all(seed)

    # Generate data
    noise_fn = parse_noise_arg(args.noise)
    x, y, y_clean = generate_data(n=args.n, noise_fn=noise_fn)

    # Initialize parameters
    W = tf.Variable([args.init_w], dtype=tf.float32)
    B = tf.Variable([args.init_b], dtype=tf.float32)

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(args.batch_size, drop_remainder=False)

    # Select loss function
    loss_fn, loss_name = make_loss_fn(args.loss)

    lr = args.lr
    best_loss = float('inf')
    no_improve = 0
    history = {"epoch": [], "loss": [], "lr": [], "W": [], "B": []}

    print(f"\nTraining started with loss={loss_name}, seed={seed}\n")

    for epoch in range(1, args.epochs + 1):
        # Optionally add noise to parameters
        if args.w_noise_std > 0:
            W.assign_add(tf.random.normal(W.shape, stddev=args.w_noise_std))
        if args.b_noise_std > 0:
            B.assign_add(tf.random.normal(B.shape, stddev=args.b_noise_std))

        # Optional noise in learning rate
        lr_effective = max(lr + np.random.normal(0, args.lr_noise_std), 1e-8) \
                       if args.lr_noise_std > 0 else lr

        # Batch training
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                yhat = W * xb + B
                loss = loss_fn(yb, yhat)
            dW, dB = tape.gradient(loss, [W, B])
            W.assign_sub(lr_effective * dW)
            B.assign_sub(lr_effective * dB)

        # Compute epoch loss
        yhat_full = W * x + B
        epoch_loss = float(loss_fn(y, yhat_full).numpy())

        # Learning rate scheduling
        if best_loss - epoch_loss < args.tol:
            no_improve += 1
        else:
            best_loss = epoch_loss
            no_improve = 0

        if args.patience and no_improve >= args.patience:
            lr = max(lr * 0.5, args.min_lr)
            no_improve = 0

        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["lr"].append(lr)
        history["W"].append(float(W.numpy()[0]))
        history["B"].append(float(B.numpy()[0]))

        if epoch % args.print_every == 0:
            print(f"Epoch {epoch:4d} | Loss={epoch_loss:.6f} | "
                  f"W={W.numpy()[0]:.3f}, B={B.numpy()[0]:.3f} | LR={lr:.4g}")

    # -------------------------------------------------------------
    # Plot and Save Results
    # -------------------------------------------------------------
    plt.figure()
    plt.plot(history["epoch"], history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({loss_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)

    plt.figure()
    plt.scatter(x, y, s=4, alpha=0.4, label="Noisy Data")
    xs = np.linspace(-5, 5, 200).astype(np.float32)
    plt.plot(xs, 3 * xs + 2, label="True f(x)=3x+2", linewidth=2)
    plt.plot(xs, float(W.numpy()[0]) * xs + float(B.numpy()[0]),
             "--", label=f"Learned ŷ={W.numpy()[0]:.3f}x+{B.numpy()[0]:.3f}")
    plt.legend()
    plt.title("Model Fit")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fit.png"), dpi=150)

    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete ✅")
    print(f"Final W = {float(W.numpy()[0]):.4f}, B = {float(B.numpy()[0]):.4f}")
    print(f"Results saved to folder: {outdir}\n")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--loss", type=str, default="l2",
                        help="l2 | l1 | huber:delta=1.0 | hybrid:alpha=0.5")
    parser.add_argument("--noise", type=str, default="std=0.5",
                        help="none | std=0.5 | uniform=-1,1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seed_name", type=str, default="Sheena")
    parser.add_argument("--init_w", type=float, default=0.0)
    parser.add_argument("--init_b", type=float, default=0.0)
    parser.add_argument("--w_noise_std", type=float, default=0.0)
    parser.add_argument("--b_noise_std", type=float, default=0.0)
    parser.add_argument("--lr_noise_std", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
