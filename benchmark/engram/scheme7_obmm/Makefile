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

BINS = smoke_test

.PHONY: all clean help

all: $(BINS)
	@echo
	@echo "Built: $(BINS)"
	@echo "Next:  sudo ./smoke_test       # single-node loopback test"
	@echo "       sudo ./smoke_test 16    # try a larger alloc (16 MB)"

smoke_test: smoke_test.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ smoke_test.c $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f $(BINS) *.o

help:
	@echo "Scheme 7 — direct obmm UAPI"
	@echo ""
	@echo "Targets:"
	@echo "  make            build smoke_test"
	@echo "  make smoke_test rebuild just the smoke test"
	@echo "  make clean      remove binaries"
	@echo ""
	@echo "First run:"
	@echo "  make clean && make"
	@echo "  sudo ./smoke_test"
