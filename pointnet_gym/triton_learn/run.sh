# You can compile the cuda infer_cu.cu to test your training model
nvcc infer_cu.cu -o infer_cu -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
./infer_cu