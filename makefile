# Makefile for the point DMK

# compiler, and linking from C, fortran
CC = gcc
CXX = g++
FC = gfortran
FAST_KER = ON

#CC = icx
#CXX = icpx
#FC = ifx

CLINK = -lstdc++
FLINK = $(CLINK)

# if need to check memory leak using valgrind, use -march=x86-64 instead
FFLAGS = -fPIC -O3 -march=native -funroll-loops -std=legacy -w
#-fsanitize=address -g -rdynamic
# -pg -no-pie is for profiling
#FFLAGS = -fPIC -O3 -march=native -funroll-loops -std=legacy -pg -no-pie -Wall
CFLAGS= -fPIC -O3 -march=native -funroll-loops -std=c99
#-fsanitize=address -g -rdynamic
CXXFLAGS= -std=c++17 -DSCTL_PROFILE=-1 -fPIC -O3 -march=native -funroll-loops 
#-fsanitize=address -g -rdynamic

ifeq ($(FAST_KER),ON)
  CXXFLAGS= -std=c++17 -DSCTL_PROFILE=-1 -fPIC -O3 -march=native -funroll-loops 
# gcc flags
#  CXXFLAGS= -std=c++11 -DSCTL_PROFILE=-1 -fPIC -O3 -march=native -funroll-loops
endif

# set linking libraries
CLIBS = -lgfortran -lm -ldl

LIBS := -lm $(CLINK)

# extra flags for multithreaded: C/Fortran, MATLAB
OMPFLAGS =-fopenmp 
OMPLIBS =-lgomp 
OMP = OFF

LBLAS = -lblas -llapack
#LBLAS = -lopenblas

#LBLAS = -qmkl=sequential


# absolute path of this makefile, ie DMK's top-level directory...
DMK = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# For your OS, override the above by placing make variables in make.inc
-include make.inc

# additional compile flags for FAST_KER
LIBS += -lstdc++
DYLIBS += -lstdc++
CLIBS += -lstdc++
CFLAGS += -lstdc++


DYLIBS = -lm
F2PYDYLIBS = -lm -lblas -llapack

# multi-threaded libs & flags, and req'd flags (OO for new interface)...
ifneq ($(OMP),OFF)
  CXXFLAGS += $(OMPFLAGS)
  CFLAGS += $(OMPFLAGS)
  FFLAGS += $(OMPFLAGS)
  MFLAGS += $(MOMPFLAGS) -DR2008OO
  OFLAGS += $(OOMPFLAGS) -DR2008OO
  LIBS += $(OMPLIBS)
endif

LIBNAME=$(PREFIX_LIBNAME)
ifeq ($(LIBNAME),)
	LIBNAME=libdmk
endif
ifeq ($(MINGW),ON)
  DYNLIB = lib/$(LIBNAME).dll
else
  DYNLIB = lib/$(LIBNAME).so
endif

DYNAMICLIB = $(LIBNAME).so
STATICLIB = $(LIBNAME).a
LIMPLIB = $(DYNAMICLIB)
# absolute path to the .so, useful for linking so executables portable...
ABSDYNLIB = $(DMK)$(DYNLIB)

LLINKLIB = $(subst lib, -l, $(LIBNAME))

#
#
LIBS += $(LBLAS) $(LDBLASINC)
DYLIBS += $(LBLAS) $(LDBLASINC)

# vectorized kernel directory
SRCDIR = ./vec-kernels/src
INCDIR = ./vec-kernels/include
#
# objects to compile
# 
# Common objects
COM = src/common
COMOBJS = $(COM)/prini_new.o \
	$(COM)/hkrand.o \
	$(COM)/dlaran.o \
	$(COM)/cumsum.o \
	$(COM)/fmmcommon2d.o \
	$(COM)/lapack_f77.o \
	$(COM)/voltab2d.o \
	$(COM)/voltab3d.o \
	$(COM)/specialfunctions/calck0.o \
	$(COM)/specialfunctions/calck1.o \
	$(COM)/specialfunctions/besk0.o \
	$(COM)/specialfunctions/besk1.o \
	$(COM)/specialfunctions/besj0.o \
	$(COM)/specialfunctions/besj1.o \
	$(COM)/specialfunctions/caljy0.o \
	$(COM)/specialfunctions/caljy1.o \
	$(COM)/specialfunctions/cdjseval2d.o \
	$(COM)/specialfunctions/hank103.o \
	$(COM)/specialfunctions/legeexps.o \
	$(COM)/specialfunctions/chebexps.o \
	$(COM)/specialfunctions/orthom.o \
	$(COM)/specialfunctions/qrsolve.o \
	$(COM)/specialfunctions/proquadr.o \
	$(COM)/specialfunctions/prolcrea.o \
	$(COM)/specialfunctions/prolaterouts.o \
	$(COM)/polytens.o \
	$(COM)/dmk_routs.o \
	$(COM)/pts_tree_per.o \
	$(COM)/tree_routs_per.o \
	$(COM)/tree_data_routs.o \
	$(COM)/tensor_prod_routs.o


COMOBJS+= $(SRCDIR)/libkernels7.o

# point DMK objects
PDMK = src/pdmk
PDMKOBJS = $(PDMK)/lndiv.o \
	$(PDMK)/lndiv_fast.o \
	$(PDMK)/kernelevaluation/st2dkernels.o \
	$(PDMK)/kernelevaluation/st3dkernels.o \
	$(PDMK)/pdmk_pwterms_stokes3.o \
	$(PDMK)/pdmk_local.o \
	$(PDMK)/kernel_FT.o \
	$(PDMK)/stokes_kernel_FT3.o \
	$(PDMK)/pdmk_pwrouts.o \
	$(PDMK)/stokesdmk7.o 

# Test objects
OBJS = $(COMOBJS) $(PDMKOBJS)


.PHONY: usage test-static test-dyn python 

default: usage

$(SRCDIR)/libkernels7.o: $(SRCDIR)/libkernels7.cpp
		$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

usage:
	@echo "-------------------------------------------------------------------------"
	@echo "Makefile for DMK. Specify what to make:"
	@echo "  make test-static - compile and run validation tests"
	@echo "  make objclean - removal all object files, preserving lib"
	@echo "  make clean - also remove lib and demo executables"
	@echo ""
	@echo "For faster (multicore) making, append the flag -j"
	@echo "-------------------------------------------------------------------------"

#
# implicit rules for objects (note -o ensures writes to correct dir)
#
%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@
%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@
%.o: %.f
	$(FC) -c $(FFLAGS) $< -o $@
%.o: %.f90 
	$(FC) -c $(FFLAGS) $< -o $@

#
# build the library...
#
#
# testing routines
#
test-static:  $(OBJS) test/pdmk-static
	cd test/pdmk; ./int2-pdmk

test/pdmk-static:
	$(FC) $(FFLAGS) test/pdmk/teststokesdmk.f -o test/pdmk/int2-pdmk $(OBJS) $(LIBS)


#
# housekeeping routines
#
clean: objclean
	rm -f test/pdmk/int2-pdmk
	rm -f test/bdmk/int2-bdmk

objclean: 
	rm -f $(OBJS)
	rm -f test/pdmk/*.o 
	rm -f test/bdmk/*.o 
