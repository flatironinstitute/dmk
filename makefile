# Makefile for the point DMK

# compiler, and linking from C, fortran
#CC = gcc
#CXX = g++
#FC = gfortran

CC = icx
CXX = icpc
FC = ifort

CLINK = -lstdc++
FLINK = $(CLINK)

# if need to check memory leak using valgrind, use -march=x86-64 instead
FFLAGS = -fPIC -O3 -march=native -funroll-loops -std=legacy -w
# -fsanitize=address -g -rdynamic
# -pg -no-pie is for profiling
#FFLAGS = -fPIC -O3 -march=native -funroll-loops -std=legacy -pg -no-pie -Wall	
CFLAGS= -fPIC -O3 -march=native -funroll-loops -std=c99
#-fsanitize=address -g -rdynamic
CXXFLAGS= -std=c++17 -DSCTL_PROFILE=-1 -fPIC -O3 -march=native -funroll-loops
#-fsanitize=address -g -rdynamic

ifeq ($(FAST_KER),ON)
  CXXFLAGS= -std=c++17 -DSCTL_PROFILE=-1 -DSCTL_HAVE_SVML -fPIC -O3 -march=native -funroll-loops
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

#LBLAS = -lblas -llapack
#LBLAS = -lopenblas

LBLAS = -qmkl=sequential

DMK_INSTALL_DIR=$(PREFIX)
ifeq ($(PREFIX),)
	DMK_INSTALL_DIR = ${HOME}/lib
endif


# absolute path of this makefile, ie DMK's top-level directory...
DMK = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# For your OS, override the above by placing make variables in make.inc
-include make.inc

# additional compile flags for FAST_KER
LIBS += -lstdc++
DYLIBS += -lstdc++
CLIBS += -lstdc++
FFLAGS += -lstdc++
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
VCLDIR = ./extern/VCL/version2
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
	$(COM)/pts_tree.o \
	$(COM)/tree_routs.o \
	$(COM)/tree_data_routs.o \
	$(COM)/tensor_prod_routs.o


COMOBJS+= $(SRCDIR)/libkernels.o

# point DMK objects
PDMK = src/pdmk
PDMKOBJS = $(PDMK)/lndiv.o \
	$(PDMK)/lndiv_fast.o \
	$(PDMK)/kernelevaluation/y2dkernels.o \
	$(PDMK)/kernelevaluation/y3dkernels.o \
	$(PDMK)/kernelevaluation/y3dkernels_fast.o \
	$(PDMK)/kernelevaluation/l3dkernels.o \
	$(PDMK)/kernelevaluation/logkernels.o \
	$(PDMK)/kernelevaluation/sl3dkernels.o \
	$(PDMK)/pdmk_pwterms.o \
	$(PDMK)/pdmk_local.o \
	$(PDMK)/kernel_FT.o \
	$(PDMK)/stokes_kernel_FT.o \
	$(PDMK)/pdmk_pwrouts.o \
	$(PDMK)/pdmk.o 

# Test objects
OBJS = $(COMOBJS) $(PDMKOBJS)



.PHONY: usage lib install test-static test-dyn python 

default: usage

$(SRCDIR)/libkernels.o: $(SRCDIR)/libkernels.cpp
		$(CXX) $(CXXFLAGS) -I$(VCLDIR) -I$(INCDIR) -c $^ -o $@

usage:
	@echo "-------------------------------------------------------------------------"
	@echo "Makefile for DMK. Specify what to make:"
	@echo "  make install - compile and install the main library"
	@echo "  make install PREFIX=(INSTALL_DIR) - compile and install the main library at custom location given by PREFIX"
	@echo "  make lib - compile the main library (in lib/ and lib-static/)"
	@echo "  make test-static - compile and run validation tests"
	@echo "  make test-dyn - test successful installation by validation tests linked to dynamic library"
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
lib: $(STATICLIB) $(DYNAMICLIB)
ifneq ($(OMP),OFF)
	@echo "$(STATICLIB) and $(DYNAMICLIB) built, multithread versions"
else
	@echo "$(STATICLIB) and $(DYNAMICLIB) built, single-threaded versions"
endif

$(STATICLIB): $(OBJS) 
	ar rcs $(STATICLIB) $(OBJS)
	mv $(STATICLIB) lib-static/

$(DYNAMICLIB): $(OBJS) 
	$(FC) -shared -fPIC $(OBJS) -o $(DYNAMICLIB) $(DYLIBS)
	mv $(DYNAMICLIB) lib/
	[ ! -f $(LIMPLIB) ] || mv $(LIMPLIB) lib/

install: $(STATICLIB) $(DYNAMICLIB)
	echo $(DMK_INSTALL_DIR)
	mkdir -p $(DMK_INSTALL_DIR)
	cp -f lib/$(DYNAMICLIB) $(DMK_INSTALL_DIR)/
	cp -f lib-static/$(STATICLIB) $(DMK_INSTALL_DIR)/
	[ ! -f lib/$(LIMPLIB) ] || cp lib/$(LIMPLIB) $(DMK_INSTALL_DIR)/
	@echo "Make sure to include " $(DMK_INSTALL_DIR) " in the appropriate path variable"
	@echo "    LD_LIBRARY_PATH on Linux"
	@echo "    PATH on windows"
	@echo "    DYLD_LIBRARY_PATH on Mac OSX (not needed if default installation directory is used"
	@echo " "
	@echo "In order to link against the dynamic library, use -L"$(DMK_INSTALL_DIR)  " "$(LLINKLIB) " -L"$(FINUFFT_INSTALL_DIR)  " "$(LFINUFFTLINKLIB)


#
# testing routines
#
test-static: $(STATICLIB)  test/pdmk-static
	cd test/pdmk; ./int2-pdmk

test-dyn: $(DYNAMICLIB)  test/pdmk-dyn
	cd test/pdmk; ./int2-pdmk

test/pdmk-static:
	$(FC) $(FFLAGS) test/pdmk/testpdmk.f -o test/pdmk/int2-pdmk lib-static/$(STATICLIB) $(LIBS) 

#
# Linking test files to dynamic libraries
#

test/pdmk-dyn:
	$(FC) $(FFLAGS) test/pdmk/testpdmk.f -o test/pdmk/int2-pdmk $(ABSDYNLIB) $(LBLAS) $(LDBLASINC)


#
# housekeeping routines
#
clean: objclean
	rm -f lib-static/*.a lib/*.so
	rm -f test/pdmk/int2-pdmk
	rm -f test/bdmk/int2-bdmk

objclean: 
	rm -f $(OBJS)
	rm -f test/pdmk/*.o 
	rm -f test/bdmk/*.o 
