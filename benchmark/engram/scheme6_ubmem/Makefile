# Scheme 6: UB Shared Memory (ubs_mem SDK) — cross-node load/store access.
#
# Prerequisites on the build host:
#   - ubs_mem package installed (Matrix Core Memory)
#       /usr/local/ubs_mem/include/ubs_mem.h
#       /usr/local/ubs_mem/include/ubs_mem_def.h
#       /usr/local/ubs_mem/lib/libubsm_sdk.so
#   - ubsmd.service running (systemctl status ubsmd)
#   - kernel modules loaded:
#       obmm, ummu, ummu_core, ubmem_vmmu, ubmempfd, ubus, ubfi
#     (install ub-pkg-mem if any are missing — it sets them up)
#
# Override UBSM_PREFIX if the SDK is installed elsewhere.

UBSM_PREFIX ?= /usr/local/ubs_mem

CC       ?= gcc
CFLAGS    = -Wall -Wextra -O2 -g -std=c11 -D_GNU_SOURCE -fPIC
INCLUDES  = -I$(UBSM_PREFIX)/include -I.
LDFLAGS   = -L$(UBSM_PREFIX)/lib -lubsm_sdk \
            -Wl,-rpath,$(UBSM_PREFIX)/lib

.PHONY: all clean

all: ubsm_probe server client

# The wrapper library source — both server and client link it statically
# via the per-target compilation below.
WRAPPER_SRC = ubmem_rw.c
WRAPPER_HDR = ubmem_rw.h

ubsm_probe: ubsm_probe.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

server: server.c $(WRAPPER_SRC) $(WRAPPER_HDR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ server.c $(WRAPPER_SRC) $(LDFLAGS)
	@echo "Built $@"

client: client.c $(WRAPPER_SRC) $(WRAPPER_HDR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ client.c $(WRAPPER_SRC) $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f ubsm_probe server client *.o
