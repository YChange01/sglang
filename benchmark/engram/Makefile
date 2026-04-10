# Engram Cross-Node Benchmark — Top-level Makefile
#
# Usage:
#   make            — build everything
#   make server     — build unified server only
#   make bench      — build benchmark client only
#   make clean      — remove all binaries

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_GNU_SOURCE -g

# ubs_mem SDK
UBSMEM_INC = -I/usr/local/ubs_mem/include
UBSMEM_LIB = -L/usr/local/ubs_mem/lib -lubsm_sdk -Wl,-rpath,/usr/local/ubs_mem/lib

# URMA SDK
URMA_INC = -I/usr/include
URMA_LIB = -L/usr/lib64 -lurma -Wl,-rpath,/usr/lib64

LIB_INC = -Ilib
COMMON_LIBS = -lpthread -lm

.PHONY: all server bench clean

all: server bench

server: server/server
bench: bench/bench_all

server/server: server/server.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(UBSMEM_INC) $(URMA_INC) $(LIB_INC) \
		-o $@ server/server.c lib/urma_rw.c \
		$(UBSMEM_LIB) $(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_all: bench/bench_all.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(UBSMEM_INC) $(URMA_INC) $(LIB_INC) \
		-o $@ bench/bench_all.c lib/urma_rw.c \
		$(UBSMEM_LIB) $(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

clean:
	rm -f server/server bench/bench_all
