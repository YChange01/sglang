# Scheme 5: URMA RW (urma_read based cross-node access)
#
# Prerequisites:
#   - umdk-urma-devel package (urma_api.h)
#   - liburma.so (typically /usr/lib64/liburma.so)

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c11 -D_GNU_SOURCE -g
INCLUDES = -I/usr/include -I.
LDFLAGS = -L/usr/lib64 -lurma -lpthread -Wl,-rpath,/usr/lib64

.PHONY: all clean

all: server client liburma_rw.so

server: server.c urma_rw.c urma_rw.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ server.c urma_rw.c $(LDFLAGS)
	@echo "Built $@"

client: client.c urma_rw.c urma_rw.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ client.c urma_rw.c $(LDFLAGS)
	@echo "Built $@"

liburma_rw.so: urma_rw.c urma_rw.h
	$(CC) $(CFLAGS) $(INCLUDES) -shared -o $@ urma_rw.c $(LDFLAGS)
	@echo "Built $@"

clean:
	rm -f server client liburma_rw.so *.o
