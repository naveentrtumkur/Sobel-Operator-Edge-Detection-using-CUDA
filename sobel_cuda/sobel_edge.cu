	/***
	 * File Name: sobel_edge.cu
	 * Description: This Program Performs Sobel edge detection operations on a .bmp, once by a
	 * serial algorithm, and once by a massively parallel CUDA algorithm.
	 */
	
	#include <stdio.h>
	#include <math.h>
	#include <stdlib.h>
	#include <stdio.h>
	#include <time.h>
	
	//C version of Code to be specified as extern,
	//because CUDA uses g++ version of compiler.
	extern "C"
	{
	#include "read_bmp.h"
	}
	
	//Define white,black and threshold values.
	#define PIXEL_BLACK 0
	#define PIXEL_WHITE 255
	#define PERCENT_BLACK_THRESHOLD 0.75
	
	//Define the thread hierarchy being used.
	#define CUDA_GRIDS 10
	#define CUDA_BLOCKS_PER_GRID 32
	#define CUDA_THREADS_PER_BLOCK 1024
	
	#define NS_PER_SEC 1000000000

	//Returns the pixel value when we pass row,coulmn value
	#define getPixelValue(row, col, dim) \
	   (((row) * (dim)) + (col))
	
	//Values required for timer calculation.
	static struct timespec serialStart;
	static struct timespec serialEnd;
	static struct timespec parallelStart;
	static struct timespec parallelEnd;

	/*
	 * Function to Display all the required information: matrix and CUDA parameters.
	 *
	 * @param inputFile -- name of the input image
	 * @param serialOutputFile -- name of the serial output image
	 * @param parallelOutputFile -- name of the parallel output image
	 * @param imageHeight -- Height of the image in pixels
	 * @param imageWidth -- Width of the image in pixels
	 */
	void DisplayParameters(char *inputFile, char *serialOutputFile, char *cudaOutputFile,
	   			int imageHeight,int imageWidth)
	{
	   printf("********************************************************************************\n");
	   printf("Serial and  CUDA Sobel edge detection.\n");
	   printf("\n");
	   printf("Input image: %s \t(Height is: %d pixels, Width is : %d pixels)\n", inputFile, imageHeight, imageWidth);
	   printf("Serial output image is: \t%s\n", serialOutputFile);
	   printf("CUDA output image is: \t%s\n", cudaOutputFile);
	   printf("\n");
	   printf("CUDA computation specifications are:\n");
	   printf("Grids =  %d grids\n", CUDA_GRIDS);
	   printf("Blocks=  %d blocks\n", CUDA_BLOCKS_PER_GRID);
	   printf("tpb= %d threads per block\n", CUDA_THREADS_PER_BLOCK);
	   printf("\n");
	   printf("*********************************************************************************\n");
	}
	

	/*
	 * Function to display information of threshold,timing and convergence results onto the screen.
	 *
	 * @param serialConvergenceThreshold
	 * @param serialConvergenceThreshold
	 */
	void DisplayResults(int serialConvergenceThreshold, int parallelConvergenceThreshold)
	{
	   printf("*******************************************************************************\n");
	   printf("Time taken for serial Sobel edge detection: %lf\n",
	      (getPixelValue(serialEnd.tv_sec, serialEnd.tv_nsec, NS_PER_SEC)
	      - getPixelValue(serialStart.tv_sec, serialStart.tv_nsec, NS_PER_SEC))
	      / ((double)NS_PER_SEC));

	   printf("Convergence Threshold: %d\n", serialConvergenceThreshold);
	   printf("\n");
	

	   printf("Time taken for CUDA Parallel Sobel edge detection: %lf\n",
	      (getPixelValue(parallelEnd.tv_sec, parallelEnd.tv_nsec, NS_PER_SEC)
	      - getPixelValue(parallelStart.tv_sec, parallelStart.tv_nsec, NS_PER_SEC))
	      / ((double)NS_PER_SEC));
	

	   printf("Convergence Threshold: %d\n", parallelConvergenceThreshold);
	   printf("********************************************************************************\n");
	  
	}
	

	/*
	 * Serial algorithm to perform Sobel edge detection on an input pixel
	 * image which is at different brightness thresholds until a certain percentage of
	 * pixels in the output pixel buffer are black(75% in our case).
	 *
	 * @param input -- input pixel buffer
	 * @param output -- output pixel buffer
	 * @param height -- height of pixel image
	 * @param width -- width of pixel image
	 * @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD(75%) of pixels are black
	 */
	int SerialSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width)
	{
	    //printf("height=%d, width =%d\n",height,width);
	    int i=0,j=0;
	    int gradientThreshold=0, blackPixelCount = 0 , boundary = height * width * 3 / 4;
	    int whitePixelCount =0;
	    //printf("value 75% = %d",boundary);
	    // Loop until blackpixel counts are less than boundary(75% black cells)
	    while(blackPixelCount < boundary)
	    {
	        gradientThreshold = gradientThreshold +1;
	        //printf("%d -- blackpix\n",blackPixelCount);	 
	        //printf("%d -- gradthresh\n",gradientThreshold);	 
	        blackPixelCount = 0;
	        for(i=1;i<(height-1);i++)
		{
		    for(j=1; j<(width-1);j++)
			{
		
		 	    double Gx = (1*input[getPixelValue(i - 1, j + 1, width)])
         				+ (-1 * input[getPixelValue(i-1, j - 1, width)])
         				+ (2 * input[getPixelValue(i, j + 1, width)])
         				+ (-2 * input[getPixelValue(i, j - 1, width)])
         				+ (1 * input[getPixelValue(i + 1, j + 1, width)])
         				+ (-1 * input[getPixelValue(i + 1, j - 1, width)]);

      			    double Gy = (1* input[getPixelValue(i - 1, j - 1, width)])
         		    + (2 * input[getPixelValue(i - 1, j, width)])
         		    + (1 * input[getPixelValue(i - 1, j + 1, width)])
        		    + (-1 * input[getPixelValue(i + 1, j - 1, width)])
        		    + (-2 * input[getPixelValue(i + 1, j, width)])
       			    + (-1 * input[getPixelValue(i + 1, j + 1, width)]);
			   
			   //Instead of squareroot, square threshold and compare directly with magnitude value 
      			   if(((Gx * Gx) + (Gy * Gy)) > (gradientThreshold * gradientThreshold))
      			   {
				//set the output value to white
         			output[getPixelValue(i,j,width)] = PIXEL_WHITE;
				whitePixelCount++;
     			   }
      			   else
      			   {
				//set the output value to black
         			output[getPixelValue(i,j,width)] = PIXEL_BLACK;
				blackPixelCount++;
      			   }
		        }
		    }
		    //printf("white=%d",whitePixelCount);
		    //printf("blck=%d",blackPixelCount);

	    }
	   //printf("%d -- gradthresh\n",gradientThreshold);	 
	   return gradientThreshold;
	}


	/*
	 * This is Parallel CUDA kernel function that performs a Sobel edge detection
	 * on a group of pixels. This kernel function is called from host's function.
	 *
	 */
	__global__ void CudaSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width, int gradientThreshold)
	{
	   int row = 0;
	   for(int i = 0; row < (height - 1); i++)
	   {
	      // Let the blockIdx increment beyond its dimension for cyclic distribution of the test pixels
	      int blockRow = (i * gridDim.x) + blockIdx.x;
	

	      // Calculate the row/col in the image buffer that this thread is on
	      row = (getPixelValue(blockRow, threadIdx.x, blockDim.x) / (width - 2)) + 1;
	      int col = (getPixelValue(blockRow, threadIdx.x, blockDim.x) % (width - 2)) + 1;
	

	      // Calculate Sobel magnitude of gradient directly, instead of using Sobel_Magnitude utility
	      double Gx = // ( * input[getPixelValue(row - 1, col - 1, width)])
	         	+ ( +1 * input[getPixelValue(row - 1, col + 1, width)])
	         	+ ( -1 * input[getPixelValue(row - 1, col - 1, width)])
	         	+ ( +2 * input[getPixelValue(row, col + 1, width)])
	        	+ ( -2 * input[getPixelValue(row, col - 1, width)])
	         	+ ( +1  * input[getPixelValue(row + 1, col + 1, width)])
	         	+ ( -1  * input[getPixelValue(row + 1, col - 1, width)]);
	

	      double Gy = ( +1 * input[getPixelValue(row - 1, col - 1, width)])
	         	+ ( +2 * input[getPixelValue(row - 1, col, width)])
	        	+ ( +1 * input[getPixelValue(row - 1, col + 1, width)])
	         	+ ( -1 * input[getPixelValue(row + 1, col - 1, width)])
	         	+ ( -2 * input[getPixelValue(row + 1, col, width)])
	         	+ ( -1 * input[getPixelValue(row + 1, col + 1, width)]);

	      if(((Gx * Gx) + (Gy * Gy)) > (gradientThreshold * gradientThreshold))
	      {
		 //set the output value to white.
	         output[getPixelValue(row, col, width)] = PIXEL_WHITE;
	      }
	      else
	      {
		 //set the output pixel value to black.
	         output[getPixelValue(row, col, width)] = PIXEL_BLACK;
	      }
	   }
	}
	

	/*
	 * Parallel algorithm to perform a Sobel edge detection on an input pixel
	 * image at different brightness thresholds until a certain percentage of
	 * pixels in the output pixel buffer are black.
	 *
	 * @param input -- input pixel buffer
	 * @param output -- output pixel buffer
	 * @param height -- height of pixel image
	 * @param width -- width of pixel image
	 * @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD(75%) of pixels are black
	 */
	__host__ int ParallelSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width)
	{
	   int numBlocks = CUDA_BLOCKS_PER_GRID;  
	   int threadsPerBlock = CUDA_THREADS_PER_BLOCK;
	   size_t imageMemSize =  height * width * sizeof(uint8_t);
	   uint8_t *deviceInputImage, *deviceOutputImage;
	
	   // Allocate device memory
	   cudaMalloc((void **)&deviceInputImage, imageMemSize);
	   cudaMalloc((void **)&deviceOutputImage, imageMemSize);

	   // Copy host input image to device
	   cudaMemcpy(deviceInputImage, input, imageMemSize, cudaMemcpyHostToDevice);
	
	   //define the device data-structures	
	   dim3 dimGrid(numBlocks);
	   dim3 dimBlock(threadsPerBlock);
	
	   //Perform Parallel Cuda Sobel edge detetction by calling the kernel.
	   int gradientThreshold, blackPixelCount = 0;
	   for(gradientThreshold = 0; blackPixelCount < (height * width * 75 / 100); gradientThreshold++)
	   {
	      // Launching the Kernel
	      CudaSobelEdgeDetection<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, height, width, gradientThreshold);

	      // Copy the device results array back to host
	      cudaMemcpy(output, deviceOutputImage, imageMemSize, cudaMemcpyDeviceToHost);

	      // Count the number of black pixels
	      blackPixelCount = 0;
	      for(int row = 1; row < (height - 1); row++)
	      {
	         for(int col = 1; col < (width - 1); col++)
	         {
	            if(output[getPixelValue(row, col, width)] == PIXEL_BLACK)
	            {
	               blackPixelCount++;
	            }
	         }
	      }
	   }
	   //Due to for loop increment, we need to decrement value and return it.
	   return gradientThreshold-1;
	}


	/*
	* Main function.
	*/
	int main(int argc, char* argv[])
	{
	   // Check for correct number of comand line args
	   if (argc != 4)
	   {
	      printf("Error:Incorrect arguments: <input_img.bmp> <serial_output_img.bmp> <cuda_output_img.bmp> Please try again..\n");
	      return 0;
	   }

	   // Open the files specified by the command line args
	   FILE *inputFile = fopen(argv[1], "rb");
	   FILE *serialOutputFile = fopen(argv[2], "wb");
	   FILE *cudaOutputFile = fopen(argv[3], "wb");

	   //check if the file is valid and can be opened.
	   if(inputFile == NULL)
	   {
	      printf("Error: %s file could not be opened for reading.", argv[1]);
	   }

	   // Read in input image and allocate space for new output image buffers
	   uint8_t *inputImage = (uint8_t *)read_bmp_file(inputFile);
	   uint8_t *serialOutputImage = (uint8_t *)malloc(get_num_pixel());
	   uint8_t *cudaOutputImage = (uint8_t *)malloc(get_num_pixel());

	   // Display GPU related information by calling this function.
	   DisplayParameters(argv[1], argv[2], argv[3], get_image_height(), get_image_width());
	
	   // Call the serial function for serial sobel edge detection.
	   printf("Performing serial Sobel edge detection.\n");
	   clock_gettime(CLOCK_REALTIME, &serialStart);
	   int serialConvergenceThreshold = SerialSobelEdgeDetection(inputImage, serialOutputImage, get_image_height(), get_image_width());
	   clock_gettime(CLOCK_REALTIME, &serialEnd);
	
	   // Call the CUDA function for Parallel sobel edge detection
	   printf("Performing CUDA parallel Sobel edge detection.\n");
	   clock_gettime(CLOCK_REALTIME, &parallelStart);
	   int parallelConvergenceThreshold = ParallelSobelEdgeDetection(inputImage, cudaOutputImage, get_image_height(), get_image_width());
	   clock_gettime(CLOCK_REALTIME, &parallelEnd);
	

	   //DisplayResults for parallel and serial ConvergenceThreshold.
	   DisplayResults(serialConvergenceThreshold,parallelConvergenceThreshold);

	   // Write output image buffers. Closes files and frees buffers.
	   write_bmp_file(serialOutputFile, serialOutputImage);
	   write_bmp_file(cudaOutputFile, cudaOutputImage);
	   
	   return 0;	
	}

