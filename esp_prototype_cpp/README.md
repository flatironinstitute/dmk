# esp_prototype_cpp

C++ prototype of the ESP (Ewald Splitting with Prolates) algorithm. Produced by Ioana Popa.

## Build & Run

From the `esp_prototype_cpp` directory:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sbatch run.sh
```