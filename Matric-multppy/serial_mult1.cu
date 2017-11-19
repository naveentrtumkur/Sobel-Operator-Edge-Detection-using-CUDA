#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Function to generate random number between 1 and 2
double randd() {
  return (double)rand() / (RAND_MAX) + 1.0;
}

//Serial function To multiply matrix with it's transpose
void multiply_serial(double *h_a,double *h_b, int dim)
{
int i,j,k;
float a, b, sum;

//Start the computation of matrix with it's transpose.
for(i=0; i<dim; i++) 
{
    for(j=0; j<dim; j++) 
    {

	sum = 0;
	for(k=0; k<dim; k++)
	{
	    a =h_a[(k *dim)+i];
	    b =h_a[k*dim+j ]; // Interchange indices to get the transpose
	    sum  = sum +( a * b);
	}
    h_b[ i * dim + j ] = sum; //Assign teh value to Matrix B's element. 
    }
}
}

/*
 * The kernel function. Runs on the device(GPU).
 * d_a  - Source matrix.
 * d_b  - Destination matrix.
 * dim  - Dimension.
 */

__global__ void multiply_device (double *d_a, double *d_b,int dim) {

//Declaration of required variables.
double a, b, sum;

//Retrive the thread and block specific information.
int i = threadIdx.x,j,k;

// Begine Matrix Computation.
for (j = blockIdx.x; j < dim; j += gridDim.x) {
sum = 0;
for(k=0; k<dim; k++) {
a =d_a[k *dim+i];
b =d_a[k*dim+j];
sum  = sum + a * b;
}
d_b[ i * dim + j ] = sum; 
}
}

//Main function which invokes serial & parallel multiplication functions.	
int main()
{
//Declare the required pointers.
double *h_a; //Pointer1 for host memory
double *h_b; //Pointer2 for host memory
double *h_c; //Pointer3 for host memory
double *d_a; //Pointer1 for device memory
double *d_b; //Pointer2 for device memory

//dim of vector
int dimA = 3;
int i,j;

//define thread hierarchy
int nblocks = 4; 
int tpb = 1024;

//allocate host and device memory.
size_t memSize;

//Define the memSize and allocate required memory.
memSize = dimA*dimA * sizeof(double);
h_a = (double*)malloc(dimA*dimA*sizeof(double));
h_b = (double*)malloc(dimA*dimA*sizeof(double));
h_c = (double*)malloc(dimA*dimA*sizeof(double));
//h_b = (double*)malloc(memSize);

//cudaMalloc((void**)&h_a,memSize);
cudaMalloc((void**)&d_a,memSize);
cudaMalloc((void**)&d_b,memSize);

//Tested the program works fine for 3*3 matrix.
//So we find many print statements.

//initialize host array
//printf("Initialising host array\n");
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {	
        *(h_a+(i * dimA) + j)=  randd();
	//printf("%lf ",  *(h_a+(i * dimA) + j));
    }
//printf("\n");
}

//printf("'\n");
//Call the Serial function to multiply Matrices.
multiply_serial(h_a,h_b,dimA);

//printf("Serial output is\n");
/*for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf("%lf ",*((h_b+i*dimA)+j));
    }
    printf("\n");
}*/

//printf("Copying to device");
//Copy contents to device
cudaMemcpy(d_a, h_a,memSize, cudaMemcpyHostToDevice);

//printf("Launching kernel");
//Launch kernel
dim3 dimGrid(nblocks);
dim3 dimBlock(tpb);
multiply_device<<<dimGrid,dimBlock>>>(d_a,d_b,dimA);

//get the output
cudaMemcpy(h_c,d_b,memSize, cudaMemcpyDeviceToHost);

// Print the ouput(i.e final result after multiplication)
/*printf("Parallel output is\n");
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf("%lf ",*(h_c+i*dimA+j));
    }
    printf("\n");
}*/

//free up the allocated memory
free(h_a);
free(h_b);
free(h_c);
//free(d_a);
//free(d_b);

return 0;
}
   
