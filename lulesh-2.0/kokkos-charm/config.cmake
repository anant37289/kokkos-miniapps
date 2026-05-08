
set(CHARM_DIR "/root/charm/netlrts-linux-x86_64-smp")
set(BASE_DIR "/root/kokkos-miniapps/lulesh-2.0/kokkos-charm/")
#set(CUDA_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8")
set(KOKKOS_DIR "/root/kokkos/install")

set(CHARMC "${CHARM_DIR}/bin/charmc")
set(CPU_OPTS "-c++-option -std=c++17 -fPIC -O3 -march=native -DNDEBUG")
set(GPU_OPTS "-fPIC -O3 -march=native -DNDEBUG")
set(GPU_LINK_OPTS -O3 -language charm++ -module EveryLB -L${KOKKOS_DIR}/lib -lkokkoscore -lkokkoscontainers -lkokkossimd -Wl,-rpath,${KOKKOS_DIR}/lib)
set(LD_OPTS "-no-pie")
set(INCS "-I${BASE_DIR}")
