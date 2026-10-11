"""Health checks for containers: exit 0 if a shard (gRPC health service) or the front desk (/health)
answers that it's serving, 1 otherwise.

    python -m recsys.healthcheck grpc localhost:50051
    python -m recsys.healthcheck http http://localhost:8000/health
"""
import sys
import urllib.request


def main(kind, target):
    try:
        if kind == "grpc":
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            with grpc.insecure_channel(target) as channel:
                reply = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=2)
            return 0 if reply.status == health_pb2.HealthCheckResponse.SERVING else 1
        urllib.request.urlopen(target, timeout=2)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
