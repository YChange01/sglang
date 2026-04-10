# UBS-MEM Smoke Test: load/store via ubs_mem SDK
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

all: server client

server: server.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

client: client.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f server client *.o
