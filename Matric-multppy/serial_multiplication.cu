#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double randd() {
  return (double)rand() / (RAND_MAX + 1.0);
}
//Kernel -- To multiply matrix with it's transpose

__global__ void multiply_serial(double **d_a,double **d_c,int mSize)
{
int i,j,k;

//start with matrix multplication calcultaion

for(i=0;i<mSize;i++)
{
    for(j=0;j<mSize;j++)
    { 
        for(k=0;k<mSize;k++)
        {
	    *(*(d_c+i)+j) = (*(*(d_a+k)+i)) * (*(*(d_a+k)+j));
	}
    }
}

}
// End of function


int main()
{

double **h_a; //Pointer for host memory
double **d_a; //Pointer for device memory
double **d_c;// anotehr pointer to device memory
int dimA = 1024*1024;
int i,j;

//define thread hierarchy

int nblocks = 1; 
int tpb = 1024;


//allocate host and device memory.
size_t memSize;

memSize = dimA * sizeof(double*);
h_a = (double**)malloc(sizeof(double*)*1024);

for(i=0;i<1024;i++)
*(h_a+i) = (double*)malloc(sizeof(double*)*1024);

cudaMalloc((void**)&d_a,memSize);
cudaMalloc((void**)&d_c,memSize);


//initialize host array
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {	
        *(*(h_a+i)+j)= randd();
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

for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf(" ",h_a[i][j]);
    }
    printf("\n");
}

}
   

