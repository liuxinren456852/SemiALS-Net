#/bin/bash
CUDA_PATH=/usr/local/cuda
$CUDA_PATH/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
# TF1.4
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -shared -fPIC -I $TF_INC -I $CUDA_PATH/include -L$TF_LIB -I$TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
