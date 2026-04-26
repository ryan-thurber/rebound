include Makefile.defs

<<<<<<< HEAD
SOURCES=rebound.c tree.c particle.c gravity.c integrator.c integrator_whfast.c integrator_whfast512.c integrator_saba.c integrator_ias15.c integrator_sei.c integrator_bs.c integrator_leapfrog.c integrator_mercurius.c integrator_trace.c integrator_eos.c boundary.c input.c binarydiff.c output.c collision.c communication_mpi.c display.c tools.c rotations.c derivatives.c simulationarchive.c glad.c integrator_janus.c transformations.c fmemopen.c server.c frequency_analysis.c
OBJECTS:=$(SOURCES:.c=.$(OBJFILEEXT))
HEADERS:=$(SOURCES:.c=.h)
ifeq ($(CUDA), 1)
SOURCES += gravity_opt.cu
OBJECTS += gravity_opt.o
HEADERS += gravity_opt.h
endif
=======
SOURCES=rebound.c tree.c particle.c gravity.c gravity_gpu_bridge.c integrator.c integrator_whfast.c integrator_whfast512.c integrator_saba.c integrator_ias15.c integrator_sei.c integrator_bs.c integrator_leapfrog.c integrator_mercurius.c integrator_trace.c integrator_eos.c boundary.c input.c binarydiff.c output.c collision.c communication_mpi.c display.c tools.c rotations.c derivatives.c simulationarchive.c glad.c integrator_janus.c transformations.c fmemopen.c server.c frequency_analysis.c
CUOBJECTS=gravity_gpu.$(OBJFILEEXT)
OBJECTS=$(SOURCES:.c=.$(OBJFILEEXT)) $(CUOBJECTS)
HEADERS=$(SOURCES:.c=.h)
>>>>>>> ian/gpu_tree_implementation

all: $(SOURCES) $(LIBREBOUND)

ifneq ($(OS),Windows_NT)
# Linux, MacOS
%.$(OBJFILEEXT): %.c $(HEADERS)
	@echo "Compiling source file $< ..."
	$(CC) -c $(OPT) $(PREDEF) -o $@ $<

<<<<<<< HEAD
%.$(OBJFILEEXT): %.cu $(HEADERS)
	@echo "Compiling source file $< ..."
	$(CC) -c $(OPT) $(PREDEF) -o $@ $<
=======
gravity_gpu.$(OBJFILEEXT): gravity_gpu.cu gravity_gpu.h
	@echo "Compiling CUDA source file $< ..."
	nvcc -Xcompiler -fPIC -c gravity_gpu.cu -o gravity_gpu.$(OBJFILEEXT)
>>>>>>> ian/gpu_tree_implementation

$(LIBREBOUND): $(OBJECTS)
	@echo ""
	@echo "Linking shared library $@ ..."
<<<<<<< HEAD
	$(CC) $(OPT) -shared $(OBJECTS) $(LIB) -o $@

	@echo ""
=======
	nvcc -shared $(OBJECTS) $(LIB) -o $@ 
	
	@echo ""        
>>>>>>> ian/gpu_tree_implementation
	@echo "The shared library $< has been created successfully."
else
# Windows
%.$(OBJFILEEXT): %.c $(HEADERS)
	@echo "Compiling source file $< ..."
	$(CC) -c $(OPT) $(PREDEF) /Fo: $@ $<

gravity_gpu.$(OBJFILEEXT): gravity_gpu.cu gravity_gpu.h
	@echo "Compiling CUDA source file $< ..."
	nvcc -c gravity_gpu.cu -o gravity_gpu.$(OBJFILEEXT)

$(LIBREBOUND): $(OBJECTS)
	@echo ""
	@echo "Linking shared library $@ ..."
	$(CC) /D_USRDLL /D_WINDLL $(OBJECTS) /link /DLL /OUT:$@

	@echo ""
	@echo "The shared library $< has been created successfully."
endif


clean:
	@echo "Removing object files *.$(OBJFILEEXT) ..."
	@-$(RM) *.$(OBJFILEEXT)
	@echo "Removing shared library $(LIBREBOUND) ..."
	@-$(RM) *.so
	@echo "Removing coverage files ..."
	@-$(RM) *.gcda
	@-$(RM) *.gcno

