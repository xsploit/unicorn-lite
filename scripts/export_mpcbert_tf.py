"""Export only MPC-BERT inference tensors from its legacy TF checkpoint.

Run this with the isolated TensorFlow conversion environment. The live bot
environment never imports TensorFlow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    tensors: dict[str, np.ndarray] = {}
    for name, _ in tf.train.list_variables(args.checkpoint):
        if not name.startswith("bert/") or "/adam_" in name:
            continue
        tensors[name] = tf.train.load_variable(args.checkpoint, name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **tensors)
    print(f"exported {len(tensors)} inference tensors to {output}")


if __name__ == "__main__":
    main()
