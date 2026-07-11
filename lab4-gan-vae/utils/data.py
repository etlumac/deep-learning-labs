import struct
import numpy as np
from pathlib import Path

def load_idx(path):
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        num_dims = magic & 0xFF

        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(num_dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)

        return data.reshape(shape)

def load_mnist(path="data"):
    p = Path(path)

    X = load_idx(p / "train-images.idx3-ubyte")
    X = X.astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)

    return X
