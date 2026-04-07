export OPENGL=0
export SERVER=0
include ../../src/Makefile.defs

NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60

.PHONY: all librebound
all: problem.c librebound
	@echo "Compiling $< ..."
	$(CCPROBLEM) -L/opt/ohpc/pub/apps/cuda/11.0/lib64 -lcudart -lcudart
	@echo ""
	@echo "Compilation successful. To run REBOUND, execute the file '$(EXEREBOUND)'."
	@echo ""

librebound:
	@echo "Compiling shared library $(LIBREBOUND) ..."
	$(MAKE) -C ../../src/
	@-$(RM) $(LIBREBOUND)
	@$(LINKORCOPYLIBREBOUND)
	@echo ""

clean:
	@echo "Cleaning up shared library $(LIBREBOUND) ..."
	$(MAKE) -C ../../src/ clean
	@echo "Cleaning up local directory ..."
	@-$(RM) $(LIBREBOUND)
	@-$(RM) $(EXEREBOUND)
