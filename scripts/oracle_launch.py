"""Keep asking Oracle Cloud for an Always Free Arm machine until it has room, then print its address.

    ~/.oci/venv/bin/python scripts/oracle_launch.py --dry-run     # look everything up, launch nothing
    nohup caffeinate -s ~/.oci/venv/bin/python scripts/oracle_launch.py > ~/.oci/launch.log 2>&1 &
    python scripts/oracle_launch.py --tries 1 --first 7            # one request (the GitHub workflow's way)

Oracle's free Arm machines (VM.Standard.A1.Flex) are often "out of host capacity". Every --every seconds
this asks for one, going round the availability domains and sizes (2 cores / 8 GB, then 1 / 6), waiting
twice as long after each "too many requests" (up to 10 minutes) and speeding up again slowly, so it settles
at a pace Oracle accepts. It stops at the first success, or
as soon as a machine with this name exists, so it never makes two. A refused request costs nothing, and
only the free shape is ever asked for. Needs Oracle's Python library
(pip install oci, kept out of the project's environment: ~/.oci/venv) and an API key in ~/.oci/config.
The network is the VCN wizard's: the public subnet of --vcn. Run by a GitHub Actions workflow
(.github/workflows/oracle-launch.yml), it hands "done" and "ip" to the workflow's next steps once the
machine exists.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import oci

SHAPE = "VM.Standard.A1.Flex"
# A dropped connection, a timeout, the network down after the Mac wakes up: waited out and retried, never fatal.
NETWORK_ERRORS = (oci.exceptions.RequestException, oci.exceptions.ConnectTimeout, ConnectionError, TimeoutError)
SIZES = [(2, 8), (1, 6)]  # (OCPUs, GB of memory), within the Always Free 2 OCPU / 12 GB


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def notify(message):
    """A macOS notification (with sound), so a success is seen even if nobody is watching the log."""
    if sys.platform != "darwin":
        return
    subprocess.run(["osascript", "-e", f'display notification "{message}" with title "Oracle" sound name "Glass"'],
                   check=False)


def lookups(args, config):
    """(availability domains, public subnet, Ubuntu image, SSH public key) for the launch."""
    tenancy = config["tenancy"]  # the root compartment
    identity, compute = oci.identity.IdentityClient(config), oci.core.ComputeClient(config)
    network = oci.core.VirtualNetworkClient(config)
    domains = [d.name for d in identity.list_availability_domains(tenancy).data]
    vcns = [v for v in network.list_vcns(tenancy).data if v.display_name == args.vcn and v.lifecycle_state == "AVAILABLE"]
    if len(vcns) != 1:
        sys.exit(f"expected one VCN named {args.vcn!r}, found {len(vcns)}")
    subnets = [s for s in network.list_subnets(tenancy, vcn_id=vcns[0].id).data
               if not s.prohibit_public_ip_on_vnic and s.lifecycle_state == "AVAILABLE"]
    if len(subnets) != 1:
        sys.exit(f"expected one public subnet in {args.vcn!r}, found {[s.display_name for s in subnets]}")
    images = [i for i in compute.list_images(tenancy, operating_system="Canonical Ubuntu", operating_system_version="24.04",
                                             shape=SHAPE, sort_by="TIMECREATED", sort_order="DESC").data
              if "Minimal" not in i.display_name]
    if not images:
        sys.exit("no Ubuntu 24.04 image for the Arm shape")
    key = Path(args.ssh_key).expanduser().read_text().strip()
    return domains, subnets[0], images[0], key


def report(**values):
    """Hand results to the GitHub Actions workflow running this (its next steps read them), if one is."""
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def existing(compute, tenancy, name):
    return [i for i in compute.list_instances(tenancy, display_name=name).data
            if i.lifecycle_state not in ("TERMINATED", "TERMINATING")]


def public_ip(config, compute, instance):
    network = oci.core.VirtualNetworkClient(config)
    for attachment in compute.list_vnic_attachments(instance.compartment_id, instance_id=instance.id).data:
        ip = network.get_vnic(attachment.vnic_id).data.public_ip
        if ip:
            return ip
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="song-recommender")
    ap.add_argument("--vcn", default="song-recommender-vcn")
    ap.add_argument("--ssh-key", default="~/.ssh/oracle_songrec.pub")
    ap.add_argument("--every", type=float, default=60, help="seconds between requests (at the fastest)")
    ap.add_argument("--tries", type=int, default=0, help="stop after this many requests (0: until it gets one)")
    ap.add_argument("--first", type=int, default=0, help="start at this (zone, size) choice, counting round")
    ap.add_argument("--dry-run", action="store_true", help="only look things up and print them")
    args = ap.parse_args()

    config = oci.config.from_file()
    oci.config.validate_config(config)
    tenancy, compute = config["tenancy"], oci.core.ComputeClient(config)
    domains, subnet, image, key = lookups(args, config)
    log(f"region {config['region']}; domains {domains}; subnet {subnet.display_name}; image {image.display_name}")
    if found := existing(compute, tenancy, args.name):
        ip = public_ip(config, compute, found[0])
        log(f"STOPPED: a machine named {args.name!r} already exists ({found[0].lifecycle_state}, IP {ip})")
        report(done="true", ip=ip)
        return
    if args.dry_run:
        log(f"dry run: would ask for {SHAPE} at sizes {SIZES}, one request every {args.every:.0f}s")
        return

    choices = [(domain, size) for domain in domains for size in SIZES]
    attempt, wait, check = 0, args.every, False
    while True:
        # Before each round, and after a network error (a launch may have gone through with only its reply lost):
        # is there a machine already?
        if check or (attempt and attempt % len(choices) == 0):
            try:
                found = existing(compute, tenancy, args.name)
            except NETWORK_ERRORS as e:
                wait = min(wait * 2, 600)
                log(f"network error checking for the machine ({type(e).__name__}); next check in {wait:.0f}s")
                time.sleep(wait + random.uniform(0, 5))
                continue
            check = False
            if found:
                log(f"STOPPED: a machine named {args.name!r} exists now ({found[0].lifecycle_state})")
                report(done="true", ip=public_ip(config, compute, found[0]))
                return
        domain, (ocpus, memory) = choices[(args.first + attempt) % len(choices)]
        attempt += 1
        details = oci.core.models.LaunchInstanceDetails(
            availability_domain=domain, compartment_id=tenancy, display_name=args.name, shape=SHAPE,
            shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(ocpus=ocpus, memory_in_gbs=memory),
            source_details=oci.core.models.InstanceSourceViaImageDetails(image_id=image.id),
            create_vnic_details=oci.core.models.CreateVnicDetails(subnet_id=subnet.id, assign_public_ip=True),
            metadata={"ssh_authorized_keys": key})
        try:  # no automatic retries by the library: this loop sets the pace
            instance = compute.launch_instance(details, retry_strategy=oci.retry.NoneRetryStrategy()).data
        except oci.exceptions.ServiceError as e:
            if e.status == 500 and "capacity" in str(e.message).lower():
                log(f"try {attempt}: {ocpus} OCPU / {memory} GB in {domain.split(':')[-1]}: out of capacity")
                wait = max(args.every, wait * 0.9)  # speed up again slowly after being told to slow down
            elif e.status == 429:  # asking too often: back off, doubling up to 10 minutes
                wait = min(wait * 2, 600)
                log(f"try {attempt}: too many requests; next try in {wait:.0f}s")
            else:
                log(f"STOPPED: unexpected error {e.status} {e.code}: {e.message}")
                notify(f"Launch script stopped: {e.code}")
                raise
            if args.tries and attempt >= args.tries:
                return
            time.sleep(wait + random.uniform(0, 5))
            continue
        except NETWORK_ERRORS as e:  # the request may or may not have reached Oracle: check before asking again
            wait, check = min(wait * 2, 600), True
            log(f"try {attempt}: network error ({type(e).__name__}); checking for the machine, next try in {wait:.0f}s")
            if args.tries and attempt >= args.tries:
                return
            time.sleep(wait + random.uniform(0, 5))
            continue
        log(f"LAUNCHED {ocpus} OCPU / {memory} GB in {domain}: {instance.id}")
        instance = oci.wait_until(compute, compute.get_instance(instance.id), "lifecycle_state", "RUNNING",
                                  max_wait_seconds=900).data
        ip = public_ip(config, compute, instance)
        log(f"RUNNING at public IP {ip}")
        notify(f"Your Oracle machine is running at {ip}")
        report(done="true", ip=ip)
        return


if __name__ == "__main__":
    main()
