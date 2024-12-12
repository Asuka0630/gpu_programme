# nvcc ./v7.cu -o v7.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

nvcc ./v1.cu -o v1.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v1: "
./v1.out ../param/default_param
echo " "
rm v1.out

nvcc ./v2.cu -o v2.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v2: "
./v2.out ../param/default_param
echo " "
rm v2.out

nvcc ./v3.cu -o v3.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v3: "
./v3.out ../param/default_param
echo " "
rm v3.out

nvcc ./v4.cu -o v4.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v4: "
./v4.out ../param/default_param
echo " "
rm v4.out

nvcc ./v6.cu -o v6.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v6: "
./v6.out ../param/default_param
echo " "
rm v6.out

nvcc ./v7.cu -o v7.out -Xcompiler "-O3 -std=c++14" -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
echo "v7: "
./v7.out ../param/8192
echo " "
rm v7.out
