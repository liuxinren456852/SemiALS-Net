#/bin/bash
/usr/local/cuda/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.12, Py3.5
CUDA_PATH=/usr/local/cuda
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I $TF_INC -I $CUDA_PATH/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -L/home/mmvc/mmvc-ad-local/mmvc-ad-local-001/miniconda3/envs/XL_py3_cuda10/lib/python3.7/site-packages/tensorflow/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
