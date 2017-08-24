export OPENCV_OPENCL_DEVICE=":ACCELERATOR:"
./ldws --config-file examples/road-dual.conf --enable-opencl
#strace -ff -o trace ./ldws --config-file examples/road-dual.conf --enable-opencl
