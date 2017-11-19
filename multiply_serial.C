#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double randd() {
  return (double)rand() / (RAND_MAX + 1.0);
}

int main()
{

int *h_a; //Pointer for host memory
int *d_a; //Pointer for device memory
int *d_c;// anotehr pointer to device memory
int dimA = 1024*1024;
int i,j;

//define thread hierarchy

int nblocks = 2
int tpb = 1024


//allocate host and device memory.
size_t memSize;

memSize = dimA * sizeof(double);
h_a = (double*)malloc(memSize);

cudaMalloc((void**)&d_a,memSize);
cudaMalloc((void**)&d_c,memSize);


//initialize host array
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        h_a[i]= randd();
    }
}

//Copy contents to device
cudaMemcpy(d_a, h_a,memSize, cudaMemcpyHostToDevice);

//Launch kernel
dim3 dimGrid(nblocks);
dim3 dimBlock(tpb);
multiply_serial<<<dimGrid,dimBlock>>>(d_a,d_c,dimA);

//get the output
cudaMemcpy(h_a, d_c,memSize, cudaMemcpyDeviceToHost);

// Print the ouput(i.e final result after multiplication)

                                                                                                                                                                                          4,0-1         Top

