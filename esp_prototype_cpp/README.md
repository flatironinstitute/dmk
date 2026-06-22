# esp_prototype_cpp

C++ prototype of the ESP (Ewald Splitting with Prolates) algorithm. Produced by Ioana Popa.

## Build & Run

The most important file is `esp.cpp`.

From the `esp_prototype_cpp` directory:

```bash
module load gcc/13.3.0
module load openmpi/4.1.8
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sbatch run.sh
```