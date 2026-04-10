# Scheme 7 — direct obmm UAPI (no libubsm_sdk.so, no libubse.so, no ubsmd)
#
# Targets:
#   make             — build smoke_test
#   make smoke_test  — single-node loopback load/store test
#   make clean       — remove binaries
#
# Prereqs on Node1/Node2:
#   - /dev/obmm + obmm kernel module loaded (from ub-pkg-mem)
#   - /usr/include/ub/obmm.h present (from kernel-headers or ub-pkg-mem devel)
#   - Root or RW on /dev/obmm and /dev/obmm_shmdev*
#
# The smoke test is fully self-contained: single C file, no custom libs,
# just raw ioctl + mmap. If this passes, scheme7 is unblocked and we can
# build cross-node server/client on top of the same call chain.

CC      ?= cc
CFLAGS  ?= -Wall -Wextra -O2 -g -std=c11 -D_GNU_SOURCE
CPPFLAGS = -I/usr/include -I.
LDFLAGS  =

BINS = smoke_test server client

.PHONY: all clean help

all: $(BINS)
	@echo
	@echo "Built: $(BINS)"
	@echo ""
	@echo "Loopback validation (first):"
	@echo "  sudo ./smoke_test             # single-node sanity check"
	@echo ""
	@echo "Cross-node benchmark:"
	@echo "  # Node1 (data server):"
	@echo "  sudo ./server                 # port 13857, 2 GB buffer"
	@echo "  # Node2 (compute client):"
	@echo "  sudo ./client 141.61.84.245 13857 10000 341 200"

smoke_test: smoke_test.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ smoke_test.c $(LDFLAGS)
	@echo "Built $@"

server: server.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ server.c $(LDFLAGS)
	@echo "Built $@"

client: client.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ client.c $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f $(BINS) *.o

help:
	@echo "Scheme 7 — direct obmm UAPI"
	@echo ""
	@echo "Targets:"
	@echo "  make             build smoke_test + server + client"
	@echo "  make smoke_test  single-node loopback"
	@echo "  make server      cross-node data provider"
	@echo "  make client      cross-node reader + 3 bench loops"
	@echo "  make clean       remove binaries"
	@echo ""
	@echo "Run order:"
	@echo "  1. sudo ./smoke_test              (local sanity)"
	@echo "  2. Node1: sudo ./server"
	@echo "     Node2: sudo ./client <node1-ip>"
