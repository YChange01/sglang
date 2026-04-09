# Build liburma_mmap.so for direct URMA memory mapping
#
# Prerequisites:
#   - umdk-urma-devel package installed (provides urma_api.h)
#   - liburma.so available (typically /usr/lib64/liburma.so)
#
# Usage:
#   make              # build shared library
#   make test_server  # build server-side test program
#   make test_client  # build client-side test program
#   make clean

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c11 -D_GNU_SOURCE

# URMA headers and libraries
URMA_INCLUDE ?= /usr/include
URMA_LIBDIR ?= /usr/lib64

# Detect UB mode
URMA_UBAGG_LIB = $(wildcard $(URMA_LIBDIR)/urma/liburma_ubagg.so)
ifneq ($(URMA_UBAGG_LIB),)
    CFLAGS += -DURMA_OVER_UB
endif

INCLUDES = -I$(URMA_INCLUDE) -I.
LDFLAGS = -L$(URMA_LIBDIR) -lurma -Wl,-rpath,$(URMA_LIBDIR)

.PHONY: all clean

all: liburma_mmap.so

liburma_mmap.so: urma_mmap.c urma_mmap.h
	$(CC) $(CFLAGS) $(INCLUDES) -shared -o $@ urma_mmap.c $(LDFLAGS)
	@echo "Built $@"

# Server-side test: registers memory, prints seg_info, waits
test_server: test_server.c urma_mmap.c urma_mmap.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ test_server.c urma_mmap.c $(LDFLAGS)

# Client-side test: imports remote segment, reads data
test_client: test_client.c urma_mmap.c urma_mmap.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ test_client.c urma_mmap.c $(LDFLAGS)

clean:
	rm -f liburma_mmap.so test_server test_client *.o
