"""Sample what the server's CPUs are doing while a timing check runs, for scripts/tail_report.py. Runs on the server
itself, outside the containers (the host's python3, standard library only):

    python3 scripts/machine_sampler.py --out runs/serving/tail/machine_<label>.jsonl --seconds 600 &

Every --every seconds, one line: t (Unix time); the machine's CPU time since boot by kind (/proc/stat's first line:
user, nice, system, idle, iowait, irq, softirq, steal, in clock ticks); how many threads are ready to run
(procs_running); how long some thread has waited for a CPU (/proc/pressure/cpu, microseconds since boot); and for each
container (by compose service: shard-0, frontdesk, ...) and Docker's own daemons, the CPU microseconds its processes
have used (its cgroup's cpu.stat) and waited for a CPU (cpu.pressure). All counters: the report takes differences.
Meanwhile <out>.events holds Docker's events for commands run inside containers (exec_start, exec_die), which is
what a health check is: a new Python process every 2 s in each container that has one.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

CGROUPS = Path("/sys/fs/cgroup")
DAEMONS = {"dockerd": "system.slice/docker.service", "containerd": "system.slice/containerd.service",
           "ssh sessions": "user.slice"}  # the check's client and this sampler run in an ssh session
STAT = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def containers():
    """{name: cgroup folder} for the running containers (compose service name, else the container's) and the
    daemons."""
    lines = subprocess.run(["docker", "ps", "-q", "--no-trunc"], capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for cid in lines:
        name = subprocess.run(["docker", "inspect", "-f",
                               '{{index .Config.Labels "com.docker.compose.service"}}|{{.Name}}', cid],
                              capture_output=True, text=True, check=True).stdout.strip()
        service, _, plain = name.partition("|")
        folder = CGROUPS / f"system.slice/docker-{cid}.scope"
        if folder.exists():
            out[service or plain.lstrip("/")] = folder
    return {**out, **{name: CGROUPS / path for name, path in DAEMONS.items() if (CGROUPS / path).exists()}}


def usage_us(folder):
    """CPU microseconds a cgroup's processes have used (cpu.stat's usage_usec)."""
    return next(int(line.split()[1]) for line in (folder / "cpu.stat").read_text().splitlines()
                if line.startswith("usage_usec "))


def waited_us(path):
    """Microseconds during which some thread was ready to run but waited for a CPU (a pressure file's "some" total)."""
    line = next(line for line in path.read_text().splitlines() if line.startswith("some "))
    return int(line.rpartition("total=")[2])


def sample(groups):
    stat = Path("/proc/stat").read_text().splitlines()
    cpu = dict(zip(STAT, map(int, stat[0].split()[1:9])))
    running = next(int(line.split()[1]) for line in stat if line.startswith("procs_running"))
    row = {"t": time.time(), "cpu": cpu, "running": running,
           "waited_us": waited_us(Path("/proc/pressure/cpu")), "groups": {}}
    for name, folder in groups.items():
        try:
            row["groups"][name] = [usage_us(folder), waited_us(folder / "cpu.pressure")]
        except OSError:  # a container that stopped meanwhile
            pass
    return row


def record_events(proc, path):
    """Docker's exec events (a health check's start and end) from `docker events` (proc) into path."""
    with open(path, "w") as f:
        for line in proc.stdout:
            e = json.loads(line)
            action = e.get("Action", "")
            if action.startswith(("exec_start", "exec_die")):
                attrs = e["Actor"]["Attributes"]
                f.write(json.dumps({"t": e["timeNano"] / 1e9, "action": action.split(":")[0],
                                    "command": action.partition(": ")[2],
                                    "container": attrs.get("com.docker.compose.service", attrs.get("name")),
                                    "exec": attrs.get("execID"), "exit": attrs.get("exitCode")}) + "\n")
                f.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, required=True, help="how long to sample")
    ap.add_argument("--every", type=float, default=0.05, help="seconds between samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = containers()
    proc = subprocess.Popen(["docker", "events", "--filter", "type=container", "--format", "{{json .}}"],
                            stdout=subprocess.PIPE, text=True)
    threading.Thread(target=record_events, args=(proc, out.with_suffix(".events.jsonl")), daemon=True).start()
    end, n = time.time() + args.seconds, 0
    with open(out, "w") as f:
        f.write(json.dumps({"groups": sorted(groups), "every": args.every}) + "\n")
        next_t = time.time()
        while time.time() < end:
            f.write(json.dumps(sample(groups)) + "\n")
            n += 1
            next_t += args.every
            time.sleep(max(0.0, next_t - time.time()))
    proc.terminate()
    print(f"{n:,} samples of {len(groups)} groups ({', '.join(sorted(groups))}) -> {out}")
