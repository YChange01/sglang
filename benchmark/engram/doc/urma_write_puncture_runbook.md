# URMA Write Puncture Runbook

This note records the known-good path for running the Engram URMA WRITE
ping-pong puncture on the two-node UB setup, plus the failure modes observed on
2026-05-12.

## Goal

Run URMA WRITE ping-pong without depending on UBS-MEM `ubsmd`.

The benchmark reports full RTT and half RTT:

```text
Node2 URMA WRITE -> Node1
Node1 URMA WRITE -> Node2
full RTT = round trip
1/2 RTT ~= one-way WRITE puncture latency
```

When comparing with vendor puncture results, focus on:

```text
1/2 RTT p50
1/2 RTT avg
1/2 RTT p99
```

## Known Node Info

- Node1 IP used by Node2 client: `141.61.84.245`
- `bonding_dev_0` was not present in `urma_admin show`.
- Node1 and Node2 both expose `udma2..udma7` after Node2 URMA recovery.
- Use `URMA_DEV=udma2` first unless `urma_admin show` says otherwise.
- Official `urma_sample` showed `RM/RTP` WRITE timeout and `RC` bind failure,
  while `RM/CTP` succeeded. Engram URMA defaults to `URMA_TP_TYPE=ctp` to match
  `/usr/bin/urma_sample -m 0 -t 1`.

## Preflight

Run on both nodes:

```bash
cd /home/g00872988/sglang/benchmark/engram
which urma_admin
urma_admin show
ls -la /sys/class/ubcore
ls -la /dev/ubcore
lsmod | grep -E 'mami|ubcore|udma|uburma|ipourma|ubagg|ubus|ummu|obmm'
```

Healthy output must include at least one `udma*` device in `urma_admin show`,
for example:

```text
udma2 UB ... ACTIVE
udma3 UB ... ACTIVE
...
```

If `urma_admin show` is empty on a node, URMA cannot run on that node.

## Node2 URMA Recovery

Observed failure:

```text
[urma_rw ERROR] No URMA device found (tried: bonding_dev_0 and alternatives)
```

Node2 initially had `/dev/ubcore/ubcore` but no `udma*` under
`/sys/class/ubcore`, and `urma_admin show` was empty. The recovery path was:

```bash
systemctl stop ubse.service
systemctl restart ubse.service
sleep 3
modprobe udma
modprobe uburma
urma_admin show
```

After this, Node2 showed `udma2..udma7` as `ACTIVE`.

Notes:

- `modprobe mami_linkcom`, `mami_mctrlq`, `mami_ubfabric`, `mami_ubdevm` may
  fail with "not found" even when those modules are already loaded. Trust
  `lsmod` and `urma_admin show`.
- If `urma_admin show` is still empty, inspect:

```bash
dmesg | grep -iE 'mami|udma|ubcore|ub_bus|ubdevm|ubfabric|obmm|ummu' | tail -120
ls -la /sys/class/ubcore
ls -la /sys/bus/ub_bus_controller/devices 2>/dev/null
```

## Do Not Use `bonding_dev_0` Unless Present

The benchmark had been launched with:

```bash
URMA_DEV=bonding_dev_0
```

but both nodes' current `urma_admin show` output did not include
`bonding_dev_0`. Use a real device name from `urma_admin show`; currently:

```bash
URMA_DEV=udma2
```

## numactl

`numactl` is optional. The runner supports:

```bash
NUMA_NODE=none
```

If an old checkout still says `numactl: command not found`, update to a commit
that includes the numactl fallback, or install `numactl`.

## Environment Variables

Set environment variables inline with the command or export them. Plain shell
assignments on separate lines are not inherited by child scripts unless they
are exported.

Good:

```bash
NUMA_NODE=none SERVER_IP=141.61.84.245 URMA_DEV=udma2 ./run_two_nodes.sh node2 urma-write --bytes 64
```

Also good:

```bash
export NUMA_NODE=none
export SERVER_IP=141.61.84.245
export URMA_DEV=udma2
./run_two_nodes.sh node2 urma-write --bytes 64
```

Bad:

```bash
NUMA_NODE=none
SERVER_IP=141.61.84.245
URMA_DEV=udma2
./run_two_nodes.sh node2 urma-write --bytes 64
```

## URMA TP Type

Observed on 2026-05-12:

```bash
/usr/bin/urma_sample -d udma2 -i 141.61.84.245 -p 13859
```

failed at the first WRITE completion, equivalent to `RM/RTP`.

```bash
/usr/bin/urma_sample -m 1 -d udma2 -i 141.61.84.245 -p 13859
```

failed to bind jetty in `RC`.

```bash
/usr/bin/urma_sample -m 0 -t 1 -d udma2 -i 141.61.84.245 -p 13859
```

succeeded. Therefore Engram URMA uses:

```bash
URMA_TP_TYPE=ctp
```

This is the default. Use `URMA_TP_TYPE=rtp` only to reproduce the failing
default sample behavior.

For `udma2`, `urma_admin show -d udma2 --whole` showed CTP on priorities 6 and
7, while priority 15 is RTP:

```text
priority  : ... 6 7 ... 15
tp_type   : ... CTP CTP ... RTP
```

Engram auto-selects the first priority matching `URMA_TP_TYPE`, so CTP uses
priority 6 by default. Because local measurements showed priority 15 can still
be faster on this setup, use `URMA_PRIORITY=N` to force an A/B test for
priorities 6, 7, and 15.

## Run 64B

Node1:

```bash
cd /home/g00872988/sglang/benchmark/engram
NUMA_NODE=none URMA_DEV=udma2 URMA_TP_TYPE=ctp \
./run_two_nodes.sh node1 urma-write --bytes 64 --iters 100000 --warmup 20000
```

Node2:

```bash
cd /home/g00872988/sglang/benchmark/engram
NUMA_NODE=none SERVER_IP=141.61.84.245 URMA_DEV=udma2 URMA_TP_TYPE=ctp \
./run_two_nodes.sh node2 urma-write --bytes 64 --iters 100000 --warmup 20000
```

## Run 512B

Node1:

```bash
cd /home/g00872988/sglang/benchmark/engram
NUMA_NODE=none URMA_DEV=udma2 URMA_TP_TYPE=ctp \
./run_two_nodes.sh node1 urma-write --bytes 512 --iters 100000 --warmup 20000
```

Node2:

```bash
cd /home/g00872988/sglang/benchmark/engram
NUMA_NODE=none SERVER_IP=141.61.84.245 URMA_DEV=udma2 URMA_TP_TYPE=ctp \
./run_two_nodes.sh node2 urma-write --bytes 512 --iters 100000 --warmup 20000
```

## TCP/URMA Read Without UBS-MEM

When `ubsmd` is unavailable, use the lightweight read server instead of the
unified server. It serves TCP and URMA READ from a local DRAM buffer and skips
all UBS-MEM initialization.

Node1:

```bash
cd /home/g00872988/sglang/benchmark/engram
git pull
make server bench

NUMA_NODE=none \
READ_SERVER_MODES=tcp,urma \
URMA_DEV=udma2 \
URMA_TP_TYPE=ctp \
./run_two_nodes.sh node1 read-server
```

Node2 TCP read:

```bash
cd /home/g00872988/sglang/benchmark/engram
git pull
make bench

NUMA_NODE=none \
SERVER_IP=141.61.84.245 \
./run_two_nodes.sh node2 read tcp --iters 1000
```

Node2 URMA read:

```bash
cd /home/g00872988/sglang/benchmark/engram
NUMA_NODE=none \
SERVER_IP=141.61.84.245 \
URMA_DEV=udma2 \
URMA_TP_TYPE=ctp \
./run_two_nodes.sh node2 read urma --iters 1000
```

For isolated testing, start Node1 with `READ_SERVER_MODES=tcp` or
`READ_SERVER_MODES=urma`.

## CQ Mode

Default wrapper behavior:

```text
URMA_WRITE_CQ_MOD=64
```

This requests one completion every 64 WRITEs. For pure posted WRITE:

```bash
URMA_WRITE_CQ_MOD=0 NUMA_NODE=none URMA_DEV=udma2 \
./run_two_nodes.sh node1 urma-write --bytes 64
```

Use the same `URMA_WRITE_CQ_MOD` on both nodes.

## Separate UBS-MEM Issue

`ubsmd.service active (running)` with:

```text
Status: "unavailable"
```

is a UBS-MEM management-plane issue. It blocks UBS-MEM read/write and the
unified read server, but it is not required for URMA WRITE ping-pong.
