# UBS-MEM Unified Benchmark
#
# Targets:
#   server    — data provider (ubs_mem shmem + TCP listener)
#   client    — original single-mode ubs_mem client
#   bench_all — unified multi-mode benchmark (local/tcp/urma/ubsmem variants)
#
# Prerequisites:
#   - ubs_mem SDK at /usr/local/ubs_mem/
#   - liburma.so at /usr/lib64/ (for URMA mode)
#   - scheme5_urma_rw built (urma_rw.c/h)

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_GNU_SOURCE -g

# ubs_mem SDK
UBSMEM_INC = -I/usr/local/ubs_mem/include
UBSMEM_LIB = -L/usr/local/ubs_mem/lib -lubsm_sdk -Wl,-rpath,/usr/local/ubs_mem/lib

# URMA SDK + scheme5 urma_rw library
URMA_RW_DIR = ../scheme5_urma_rw
URMA_INC = -I$(URMA_RW_DIR) -I/usr/include
URMA_LIB = -L/usr/lib64 -lurma -Wl,-rpath,/usr/lib64

COMMON_LIBS = -lpthread -lm

.PHONY: all clean

all: server client bench_all

server: server.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) -o $@ $< $(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

client: client.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) -o $@ $< $(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench_all: bench_all.c $(URMA_RW_DIR)/urma_rw.c $(URMA_RW_DIR)/urma_rw.h
	$(CC) $(CFLAGS) $(UBSMEM_INC) $(URMA_INC) -o $@ bench_all.c $(URMA_RW_DIR)/urma_rw.c \
		$(UBSMEM_LIB) $(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

clean:
	rm -f server client bench_all *.o
