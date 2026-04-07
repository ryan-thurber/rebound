export OPENGL=0
export SERVER=0
include ../../src/Makefile.defs

.PHONY: all librebound clean

all: problem.c librebound
	@echo "Compiling problem.c and linking..."
	$(CC) -std=c99 -D_GNU_SOURCE -Drestrict=__restrict__ -I../../src/ problem.c -o $(EXEREBOUND) -L. -lrebound -L/opt/ohpc/pub/apps/cuda/11.0/lib64 -lcudart -lm
	@echo "Done. Run with ./rebound"

librebound:
	@echo "Compiling shared library in ../../src/ ..."
	$(MAKE) -C ../../src/
	@-$(RM) $(LIBREBOUND)
	@$(LINKORCOPYLIBREBOUND)

clean:
	$(MAKE) -C ../../src/ clean
	@-$(RM) $(LIBREBOUND)
	@-$(RM) $(EXEREBOUND)
