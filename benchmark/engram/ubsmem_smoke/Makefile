# UBS-MEM Unified Benchmark
#
# Targets:
#   server    — data provider (ubs_mem shmem + TCP listener)
#   client    — original single-mode ubs_mem client
#   bench_all — unified multi-mode benchmark (local/tcp/ubsmem/ubsmem-nc/ubsmem-huge)
#
# Prerequisites:
#   - ubs_mem SDK installed at /usr/local/ubs_mem/
#   - ubse.service and ubsmd.service running
#   - obmm kernel module loaded

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_GNU_SOURCE -g
INCLUDES = -I/usr/local/ubs_mem/include
LDFLAGS = -L/usr/local/ubs_mem/lib -lubsm_sdk -lpthread -lm \
          -Wl,-rpath,/usr/local/ubs_mem/lib

.PHONY: all clean

all: server client bench_all

server: server.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

client: client.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

bench_all: bench_all.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f server client bench_all *.o
