"""numpy arrays in and out of gRPC messages (recsys/proto/shard.proto's Array): raw bytes plus dtype
and shape. Only plain numeric data is accepted, so a message can never carry code (unlike pickle).
Imports numpy and the generated messages only, never PyTorch: shards load FAISS, which can't share a
process with PyTorch (see recsys/ann.py)."""
import numpy as np

from recsys.proto import shard_pb2

DTYPES = {"float32", "int64", "int32", "int16"}


def pack(array):
    """An Array message holding a numpy array."""
    a = np.ascontiguousarray(array)
    if a.dtype.name not in DTYPES:
        raise TypeError(f"can't send {a.dtype} arrays; allowed: {sorted(DTYPES)}")
    return shard_pb2.Array(data=a.tobytes(), dtype=a.dtype.name, shape=list(a.shape))


def unpack(message):
    """The numpy array an Array message holds (read-only, sharing the message's bytes)."""
    if message.dtype not in DTYPES:
        raise TypeError(f"refusing an array of dtype {message.dtype!r}; allowed: {sorted(DTYPES)}")
    return np.frombuffer(message.data, dtype=message.dtype).reshape(tuple(message.shape))
