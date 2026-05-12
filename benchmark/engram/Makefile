# Engram Cross-Node Benchmark — Top-level Makefile
#
# Usage:
#   make            — build everything
#   make server     — build unified server only
#   make bench      — build benchmark clients only
#   make clean      — remove all binaries

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_GNU_SOURCE -g
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
LS64_CFLAGS ?= -march=armv8.7-a
else ifeq ($(ARCH),arm64)
LS64_CFLAGS ?= -march=armv8.7-a
else
LS64_CFLAGS ?=
endif

# ubs_mem SDK
UBSMEM_INC = -I/usr/local/ubs_mem/include
UBSMEM_LIB = -L/usr/local/ubs_mem/lib -lubsm_sdk -Wl,-rpath,/usr/local/ubs_mem/lib

# URMA SDK
URMA_INC = -I/usr/include
URMA_LIB = -L/usr/lib64 -lurma -Wl,-rpath,/usr/lib64

LIB_INC = -Ilib
COMMON_LIBS = -lpthread -lm
READ_BENCH_BINS = \
	bench/bench_local \
	bench/bench_tcp \
	bench/bench_urma_read \
	bench/bench_ubsmem_cache \
	bench/bench_ubsmem_noncache \
	bench/bench_ubsmem_import_nc \
	bench/bench_ubsmem_huge

.PHONY: all server bench clean

all: server bench

server: server/server server/read_server
bench: $(READ_BENCH_BINS) bench/ubsmem_nc_write_pingpong bench/urma_write_pingpong

server/server: server/server.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(UBSMEM_INC) $(URMA_INC) $(LIB_INC) \
		-o $@ server/server.c lib/urma_rw.c \
		$(UBSMEM_LIB) $(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

server/read_server: server/read_server.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(URMA_INC) $(LIB_INC) \
		-o $@ server/read_server.c lib/urma_rw.c \
		$(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_local: bench/bench_local.c
	$(CC) $(CFLAGS) -o $@ bench/bench_local.c $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_tcp: bench/bench_tcp.c
	$(CC) $(CFLAGS) -o $@ bench/bench_tcp.c $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_urma_read: bench/bench_urma_read.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(URMA_INC) $(LIB_INC) \
		-o $@ bench/bench_urma_read.c lib/urma_rw.c \
		$(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_ubsmem_cache: bench/bench_ubsmem_cache.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) \
		-o $@ bench/bench_ubsmem_cache.c \
		$(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_ubsmem_noncache: bench/bench_ubsmem_noncache.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) \
		-o $@ bench/bench_ubsmem_noncache.c \
		$(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_ubsmem_import_nc: bench/bench_ubsmem_import_nc.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) \
		-o $@ bench/bench_ubsmem_import_nc.c \
		$(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/bench_ubsmem_huge: bench/bench_ubsmem_huge.c
	$(CC) $(CFLAGS) $(UBSMEM_INC) \
		-o $@ bench/bench_ubsmem_huge.c \
		$(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/ubsmem_nc_write_pingpong: bench/ubsmem_nc_write_pingpong.c
	$(CC) $(CFLAGS) $(LS64_CFLAGS) $(UBSMEM_INC) \
		-o $@ bench/ubsmem_nc_write_pingpong.c \
		$(UBSMEM_LIB) $(COMMON_LIBS)
	@echo "Built $@"

bench/urma_write_pingpong: bench/urma_write_pingpong.c lib/urma_rw.c lib/urma_rw.h
	$(CC) $(CFLAGS) $(URMA_INC) $(LIB_INC) \
		-o $@ bench/urma_write_pingpong.c lib/urma_rw.c \
		$(URMA_LIB) $(COMMON_LIBS)
	@echo "Built $@"

clean:
	rm -f server/server server/read_server $(READ_BENCH_BINS) bench/ubsmem_nc_write_pingpong bench/urma_write_pingpong
