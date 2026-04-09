# Scheme 7 — direct obmm UAPI (no libubsm_sdk.so)
#
# Targets:
#   make             — build smoke_test, server, client (as they land)
#   make smoke_test  — single-node loopback sanity check
#   make clean
#
# Prereqs (on Node1/Node2):
#   - /dev/obmm exists (provided by the obmm kernel module from ub-pkg-mem)
#   - /usr/include/ub/obmm.h present (installed with the kernel headers
#     package — GPL-2.0+ UAPI header, 186 lines)
#   - Read/write access to /dev/obmm (usually root, or a group with
#     appropriate ACL)
#
# We deliberately do NOT link against libubsm_sdk.so — the entire point
# of scheme 7 is to bypass the closed userspace SDK.

CC      ?= cc
CFLAGS  ?= -Wall -Wextra -O2 -g -std=c11 -D_GNU_SOURCE -fPIC
CPPFLAGS = -I/usr/include -I.
LDFLAGS  =

LIB_SRCS = obmm_rw.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

BINS = smoke_test

.PHONY: all clean help

all: $(BINS)
	@echo
	@echo "Built: $(BINS)"
	@echo "Next:  ./smoke_test"

smoke_test: smoke_test.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ smoke_test.c $(LIB_OBJS) $(LDFLAGS)
	@echo "Built $@"

%.o: %.c obmm_rw.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(BINS) *.o

help:
	@echo "Scheme 7 obmm direct UAPI"
	@echo ""
	@echo "Targets:"
	@echo "  make            build all"
	@echo "  make smoke_test single-node loopback sanity check"
	@echo "  make clean      remove binaries and objects"
	@echo ""
	@echo "First run:"
	@echo "  make clean && make"
	@echo "  sudo ./smoke_test       # /dev/obmm usually needs root"
