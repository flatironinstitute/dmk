# Makefile for DMK
# This is the only makefile; there are no makefiles in subdirectories.
# The preferred build mechanism is cmake, but this is here for convenience

# compiler, and linking from C, fortran
CXX = g++
CXXFLAGS= -std=c++17 -fPIC -O2 -march=native -funroll-loops -Iinclude -Iextern/SCTL/include

# set linking libraries
LIBS := 

# extra flags for multithreaded: C/Fortran, MATLAB
OMPFLAGS =-fopenmp 
LBLAS = -lopenblas

LIBNAME=libdmk

DYNAMICLIB = $(LIBNAME).so
STATICLIB = $(LIBNAME).a

OBJS = src/dmk.o

.PHONY: usage lib

default: usage

usage:
	@echo "-------------------------------------------------------------------------"
	@echo "Makefile for DMK. Specify what to make:"
	@echo "  make lib - compile the main library (in lib/ and lib-static/)"
	@echo "  make objclean - remove all object files, preserving lib"
	@echo "  make clean - also remove lib"
	@echo ""
	@echo "For faster (multicore) making, append the flag -j"
	@echo "-------------------------------------------------------------------------"

#
# implicit rules for objects (note -o ensures writes to correct dir)
#
%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

#
# build the library...
#
lib: $(STATICLIB) $(DYNAMICLIB)
	@echo "$(STATICLIB) and $(DYNAMICLIB) built"

$(STATICLIB): $(OBJS) 
	ar rcs $(STATICLIB) $(OBJS)
	mkdir -p lib-static
	mv $(STATICLIB) lib-static/

$(DYNAMICLIB): $(OBJS) 
	$(CXX) -shared -fPIC $(OBJS) -o $(DYNAMICLIB) $(DYLIBS)
	mkdir -p lib
	mv $(DYNAMICLIB) lib/

#
# housekeeping routines
#
clean: objclean
	rm -f lib-static/*.a lib/*.so

objclean: 
	rm -f $(OBJS)
