"""Regenerate recsys/proto/shard_pb2.py and shard_pb2_grpc.py from recsys/proto/shard.proto.

    python scripts/gen_proto.py

Run from anywhere; protoc runs with the project root as the include path, so the generated code
imports `recsys.proto.shard_pb2` (a package path) rather than a top-level `shard_pb2`.
"""
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
code = protoc.main(["protoc", f"-I{ROOT}", f"--python_out={ROOT}", f"--grpc_python_out={ROOT}",
                    str(ROOT / "recsys" / "proto" / "shard.proto")])
raise SystemExit(code)
