#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double randd() {
  return (double)rand() / (RAND_MAX) + 1.0;
}

//To multiply matrix with it's transpose
void multiply_serial(double *h_a,double *h_b, int dim)
{
int i,j,k;

float a, b, sum;
for(i=0; i<dim; i++) 
{
    for(j=0; j<dim; j++) 
    {

	sum = 0;
	for(k=0; k<dim; k++)
	{
	    a =h_a[(k *dim)+i];
	    b =h_a[k*dim+j ];
	    sum  = sum +( a * b);
	}
    h_b[ i * dim + j ] = sum; 
    }
}
}

/*
 * The kernel function. Runs on the device.
 * d_a  - Source matrix.
 * d_b  - Destination matrix.
 * dim  - Dimension.
 */

__global__ void multiply_device (double *d_a, double *d_b,int dim) {

double a, b, sum;
int i = threadIdx.x,j,k;
for (j = blockIdx.x; j < dim; j += gridDim.x) {
sum = 0;
for(k=0; k<dim; k++) {
a =d_a[k *dim+i];
b =d_a[k*dim+j];
sum  = sum + a * b;
}
d_b[ i * dim + j ] = sum; 
}




	
	/*int k=0,i1,j1;
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index =  i * dim + j;
	double sum = 0;

	for ( k = 0; k < dim; k++) {
		sum = sum +  d_a[k * dim+ i ] * d_a[ k * dim + j ];
	}

	d_b[index] = sum;
*/
/*for(i1=0;i1<dim;i1++)
{
    for(j1=0;j1<dim;j1++)
    {
        printf(" ",*(d_b+i1*dim+j1));
    }
    printf("\n");
}
*/
 
}

//start with matrix multplication calcultaion

/*for(i=0;i<mSize;i++)
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
*/
// End of function


int main()
{

double *h_a; //Pointer1 for host memory
double *h_b; //Pointer2 for host memory
double *h_c; //Pointer3 for host memory
double *d_a; //Pointer1 for device memory
double *d_b; //Pointer2 for device memory
int dimA = 1024;
int i,j;

//define thread hierarchy
int nblocks = 4; 
int tpb = 1024;


//allocate host and device memory.
size_t memSize;

memSize = dimA*dimA * sizeof(double);
h_a = (double*)malloc(dimA*dimA*sizeof(double));
h_b = (double*)malloc(dimA*dimA*sizeof(double));
h_c = (double*)malloc(dimA*dimA*sizeof(double));
//h_b = (double*)malloc(memSize);

//for(i=0;i<1024;i++)
//*(h_a+i) = (double*)malloc(sizeof(double*)*1024);

//cudaMalloc((void**)&h_a,memSize);
cudaMalloc((void**)&d_a,memSize);
cudaMalloc((void**)&d_b,memSize);


//initialize host array
printf("Initialising host array\n");
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {	
        *(h_a+(i * dimA) + j)=  randd();
	printf("%lf ",  *(h_a+(i * dimA) + j));
    }
printf("\n");
}

printf("'\n");
multiply_serial(h_a,h_b,dimA);

printf("Serial output is\n");
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf("%lf ",*((h_b+i*dimA)+j));
    }
    printf("\n");
}

printf("Copying to device");
//Copy contents to device
cudaMemcpy(d_a, h_a,memSize, cudaMemcpyHostToDevice);

printf("Launching kernel");
//Launch kernel
dim3 dimGrid(nblocks);
dim3 dimBlock(tpb);
multiply_device<<<dimGrid,dimBlock>>>(d_a,d_b,dimA);

//get the output
cudaMemcpy(h_c,d_b,memSize, cudaMemcpyDeviceToHost);

// Print the ouput(i.e final result after multiplication)
printf("Parallel output is\n");
for(i=0;i<dimA;i++)
{
    for(j=0;j<dimA;j++)
    {
        printf("%lf ",*(h_c+i*dimA+j));
    }
    printf("\n");
}
//free(h_a);
//free(d_a);
//free(d_b);

return 0;
}
   
