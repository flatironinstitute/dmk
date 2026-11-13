
EXEC = int2-bdmk

#HOST = osx
#HOST=linux-gfortran
HOST=linux-ifort

ifeq ($(HOST),osx)
FC = gfortran
#FFLAGS = -O3 -march=native -std=legacy --openmp -funroll-loops -c -w
FFLAGS = -fPIC -O3 -march=native -std=legacy -funroll-loops -c -w
FLINK = gfortran -w -o $(EXEC)
FEND = -lopenblas ${LDFLAGS}
FEND = -L/opt/intel/oneapi/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm -ldl
#FEND = -lopenblas -Wl,--no-as-needed 
endif

ifeq ($(HOST),linux-gfortran)
FC = gfortran
FFLAGS = -fPIC -O3 -march=native -std=legacy -funroll-loops -c -w
FLINK = gfortran -w -o $(EXEC)
FEND = -L/opt/intel/oneapi/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm -ldl
endif

ifeq ($(HOST),linux-ifort)
FC = ifort
FFLAGS = -fPIC -O3 -march=native -funroll-loops -c -w
FLINK = ifort -o $(EXEC)
FEND = -qmkl=sequential
endif

SRC = ../../src


.PHONY: all clean list

SOURCES =  testbdmk.f \
  $(SRC)/common/specialfunctions/besseljs3d.f \
  $(SRC)/common/specialfunctions/hank103.f \
  $(SRC)/common/specialfunctions/legeexps.f \
  $(SRC)/common/specialfunctions/chebexps.f \
  $(SRC)/common/prini_new.f \
  $(SRC)/common/fmmcommon2d.f \
  $(SRC)/common/lapack_f77.f \
  $(SRC)/common/cumsum.f \
  $(SRC)/common/hkrand.f \
  $(SRC)/common/dlaran.f \
  $(SRC)/common/voltab2d.f \
  $(SRC)/common/voltab3d.f \
  $(SRC)/common/polytens.f \
  $(SRC)/common/tree_data_routs.f \
  $(SRC)/common/tensor_prod_routs.f \
  $(SRC)/common/pts_tree.f \
  $(SRC)/common/tree_routs.f \
  $(SRC)/common/tree_vol_coeffs.f \
  $(SRC)/common/dmk_routs.f \
  $(SRC)/bdmk/sogapproximation/get_sognodes.f \
  $(SRC)/bdmk/sogapproximation/l2dsognodes.f \
  $(SRC)/bdmk/sogapproximation/l3dsognodes.f \
  $(SRC)/bdmk/sogapproximation/sl3dsognodes.f \
  $(SRC)/bdmk/sogapproximation/y2dsognodes.f \
  $(SRC)/bdmk/sogapproximation/y3dsognodes.f \
  $(SRC)/bdmk/bdmk_local_tables.f \
  $(SRC)/bdmk/bdmk_local.f \
  $(SRC)/bdmk/bdmk_pwterms.f \
  $(SRC)/bdmk/bdmk_pwrouts.f \
  $(SRC)/bdmk/boxfgt_md.f \
  $(SRC)/bdmk/bdmk.f 

ifeq ($(WITH_SECOND),1)
SOURCES += $(SRC)/second-r8.f
endif

OBJECTS = $(patsubst %.f,%.o,$(patsubst %.f90,%.o,$(SOURCES)))

#
# use only the file part of the filename, then manually specify
# the build location
#
%.o : %.f
	$(FC) $(FFLAGS) $< -o $@

%.o : %.f90
	$(FC) $(FFLAGS) $< -o $@

%.mod : %.f90
	$(FC) $(FFLAGS) $< 


all: $(OBJECTS)
	rm -f $(EXEC)
	$(FLINK) $(OBJECTS) $(FEND)
	./$(EXEC)

clean:
	rm -f $(OBJECTS)
	rm -f $(EXEC)
	rm -f fort*
	rm -f int*

list: $(SOURCES)
	$(warning Requires:  $^)


