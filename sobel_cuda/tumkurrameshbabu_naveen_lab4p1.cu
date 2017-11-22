#include<stdio.h>
#include<stdlib.h>
#include<math.h>

//define grids and thread hierarchy
#define grids 1
#define nblocks 1024 
#define tpb 1024 

//Define the threshold value
#define Threshold 0.0001

// Function to generate random number between 1 and 2
double randd() {
  return (double)rand() / (RAND_MAX) + 1.0;
}

//Serial function To multiply matrix with it's transpose
void multiply_serial(double *h_a,double *h_b, int dim)
{
int i,j,k;
double a, b, sum;

//Start the computation of matrix with it's transpose.
for(i=0; i<dim; i++) 
{
    for(j=0; j<dim; j++) 
    {

	sum = 0;
	for(k=0; k<dim; k++)
	{
	    a =h_a[k *dim+i];
	    b =h_a[k*dim+j ]; // Interchange indices to get the transpose
	    sum  = sum +  a * b;
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

// Compare the two arrays to check they are equal, if not equal it returns the first mismatch value.
size_t CompareArrays(double *first, double *second, size_t size, double threshold)
{
   int i = 0, j = 0, index = 0;
   for(i = 0; i < size; i++)
   {
	for(j = 0 ; j < size; j++)
	{
	    index = (i * size) + j;   
      	    double diff = first[index] - second[index];
      	    // if the difference value is greater than the threshold value defined, return the mismatch index/address.
      	    if((diff > 0 && diff > threshold) || (diff < 0 && diff < (threshold * -1)))
      	    {  
		printf("value == %lfdiff",diff);
         	return index+1;
      	    }
        }
    }
   // If there is no mismatch, then return 0.
   return 0;
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

//Declaring tiem variables required.
struct timespec start, end;
struct timespec start_cuda, end_cuda;

//dim of vector
int dimA = 1024;
int i,j;


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

printf("****************************************************\n");
// Call serial version.
clock_gettime(CLOCK_REALTIME,& start);

//printf("'\n");
//Call the Serial function to multiply Matrices.
multiply_serial(h_a,h_b,dimA);

clock_gettime(CLOCK_REALTIME,& end);
double time_taken_serial = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) - \
	    		   ((double)start.tv_sec + 1.0e-9*start.tv_nsec);
//printf("Serial output is\n");
/*for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf("%lf ",*((h_b+i*dimA)+j));
    }
    printf("\n");
}*/

// call the cuda version.
clock_gettime(CLOCK_REALTIME,&start_cuda);

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

clock_gettime(CLOCK_REALTIME,&end_cuda);
	
double time_taken_cuda = ((double)end_cuda.tv_sec + 1.0e-9*end_cuda.tv_nsec) - \
		         ((double)start_cuda.tv_sec + 1.0e-9*start_cuda.tv_nsec);

printf("CUDA compute parameters are:\n");
printf("GPU with %d grid,",grids);
printf("with %d blocks ",nblocks);
printf("and with %d threads per block\n",tpb);
printf("\n");
//Print the result
printf("Size of the matrix: %d", dimA); 
printf(" x %d \n", dimA); 
printf("Time taken by the serial version:%lf", time_taken_serial);
printf(" seconds\n");
printf("Time taken by the CUDA version:%lf", time_taken_cuda);
printf("seconds\n");
printf("\n**************************************************\n");

// Code to check whether the two matrices are equivalent or not.
printf("Checking that the two resulting matrices are equivalent.\n");
size_t differ = CompareArrays(h_b, h_c, dimA, Threshold);
if(0 == differ)
{
    printf("The two resulting matrices are equivalent\n");
}

//if the two resulting matrices do not match, print the first mismatch.
else
{
    size_t badRow = (differ - 1) / dimA;
    size_t badCol = (differ - 1) % dimA;
    printf("The resulting matrices do not match!\n");
    printf("The first non-matching values occur at (%d, %d).\n", badRow, badCol);
    printf("Value of Serial[%d][%d] = %d\n", badRow, badCol, h_b[differ-1]);
    printf("Value of Parallel[%d][%d] = %d\n", badRow, badCol, h_c[differ-1]);
}
printf("\n");

//DisplayResults(h_b, h_c, MATRIX_DIM);
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
   
