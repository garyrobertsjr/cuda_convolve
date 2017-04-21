#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "sys/time.h"
#include "cuda.h"
#include "image_template.h"

__global__
void convolve_hor(float *image, int width, int height, float *mask, int mask_width, float* out_image){
	
	int i, j, k;
	float sum;
	int offsetj;
	extern __shared__ float AShared[];

	i=blockIdx.x*blockDim.x + threadIdx.x;
	j=blockIdx.y*blockDim.y + threadIdx.y;

	AShared[threadIdx.x * blockDim.x + threadIdx.y] = image[i*width+j]; 
	__syncthreads();
	sum=0;
	for(k=0; k<mask_width; k++){
		offsetj = -1*(mask_width/2)+k;
		if(threadIdx.y + offsetj >= 0 && threadIdx.y + offsetj < blockDim.y){
			sum += AShared[threadIdx.x*blockDim.x + threadIdx.y + offsetj]*mask[k];
		}
		else{
			if(blockIdx.y >=1 && blockIdx.y < gridDim.y-1)
				sum += image[i*width + (blockIdx.y * blockDim.y + threadIdx.y + offsetj)]*mask[k];
			else
				sum+=0;
		}
	}
	out_image[i*width+j] = sum;
}

__global__
void convolve_ver(float *image, int width, int height, float *mask, int mask_width, float* out_image){
	
	int i, j, k;
	float sum;
	int offseti;
	extern __shared__ float AShared[];

	i=blockIdx.x*blockDim.x + threadIdx.x;
	j=blockIdx.y*blockDim.y + threadIdx.y;

	AShared[threadIdx.x * blockDim.x + threadIdx.y] = image[i*width+j]; 
	__syncthreads();
	sum=0;
	for(k=0; k<mask_width; k++){
		offseti = -1*(mask_width/2)+k;
		if(threadIdx.x + offseti >= 0 && threadIdx.x + offseti < blockDim.x){
			sum += AShared[(threadIdx.x+offseti)*blockDim.x + threadIdx.y]*mask[k];
		}
		else{
			if(blockIdx.x >=1 && blockIdx.x < gridDim.x-1)
				sum += image[(i+offseti)*width + (blockIdx.y * blockDim.y + threadIdx.y)]*mask[k];
			else
				sum+=0;
		}
	}
	out_image[i*width+j] = sum;
}

void create_gaussians(float **g_kernel, float **dg_kernel, float sigma, int *w){
	float a = ceil(2.5*sigma-0.5);
	int sum = 0;
	
	*w=2*a+1;
	*g_kernel=(float*)malloc(sizeof(float)*(*w));

	// Calculate gaussian	
	for(int i=0; i<(*w); i++){
		(*g_kernel)[i] = exp((-1*(i-a)*(i-a))/
			  (2*sigma*sigma));
		sum+=(*g_kernel)[i];			   
	}
	
	// Normalize
	for(int i=0; i<(*w); i++){
		(*g_kernel)[i]/=sum;	
	}

	// Calculate Derivative
	sum = 0;
	
	*dg_kernel=(float*)malloc(sizeof(float)*(*w));
	
	for(int i=0; i<(*w); i++){
		(*dg_kernel)[i] = (-1*(i-a))*exp((-1*(i-a)*(i-a))/
			  (2*sigma*sigma));
		sum-=i*(*dg_kernel)[i];			   
	}
	
	// Normalize
	for(int i=0; i<(*w); i++){
		(*dg_kernel)[i]/=sum;	
	}
	
}

void print_matrix(float *matrix, int height, int width){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			printf("%.3f ", *(matrix+(i*width)+j));
		}
		printf("\n");
	}
}

int main(int argc, char **argv){
	if(argc != 3)
		printf("convolution <file> <sigma>\n");
	else{
		int height, width, k_width;
		struct timeval start, end;
		
		// CPU buffer for orig_img
		float *org_img;

		// GPU device buffer for original img
		float *d_org_img;
	
		//CPU host buffers for the final output
		float *vertical_gradient, *horizontal_gradient, *temp_gradient, *temp_vertical;

		//GPU host buffers for the final output
		float *d_vertical_gradient, *d_horizontal_gradient;
		
		// GPU buffers to hold intermediate convolution results
		float *d_temp_horizontal, *d_temp_vertical;

		// CPU host buffers to store convolution masks
		float *gaussian_kernel, *gaussian_deriv;

		// GPU device buffers to store the convolutions masks
		float *d_gaussian_kernel, *d_gaussian_deriv;
		
		read_image_template(argv[1],
				    &org_img,
				    &width,
				    &height);
		
		create_gaussians(&gaussian_kernel, &gaussian_deriv, atof(argv[2]), &k_width);
	
		printf("Gaussian Kernel:\n");
		print_matrix(gaussian_kernel, 1, k_width);
		printf("Derivative Kernel:\n");
		print_matrix(gaussian_deriv,1,k_width);

		// CPU host mallocs for GPU buffers
		cudaMalloc((void**)&d_org_img, sizeof(float)*width*height);
		cudaMalloc((void**)&d_temp_horizontal, sizeof(float)*width*height);
		cudaMalloc((void**)&d_temp_vertical, sizeof(float)*width*height);
		cudaMalloc((void**)&d_horizontal_gradient, sizeof(float)*width*height);
		cudaMalloc((void**)&d_vertical_gradient, sizeof(float)*width*height);
		cudaMalloc((void**)&d_gaussian_kernel, sizeof(float)*k_width);
		cudaMalloc((void**)&d_gaussian_deriv, sizeof(float)*k_width);

		gettimeofday(&start, NULL);
		// Offload all of the data to GPU device for convolution
		cudaMemcpy(d_org_img, org_img, sizeof(float)*width*height, cudaMemcpyHostToDevice);
		cudaMemcpy(d_gaussian_kernel, gaussian_kernel, sizeof(float)*k_width, cudaMemcpyHostToDevice);
		cudaMemcpy(d_gaussian_deriv, gaussian_deriv, sizeof(float)*k_width, cudaMemcpyHostToDevice);

		// Horizonal gradient
		int block_dim = 16;
		dim3 dmGrid(ceil(height/block_dim), ceil(width/block_dim), 1);
		dim3 dmBlock(block_dim, block_dim, 1);

		// Vertical Gradient
		convolve_hor<<<dmGrid,dmBlock, sizeof(float)*16*(16+2*floor(k_width/2))>>>
				(d_org_img, width, height, d_gaussian_kernel, k_width, d_temp_vertical);	
		convolve_ver<<<dmGrid,dmBlock,sizeof(float)*16*(16+2*floor(k_width/2))>>>
				(d_temp_vertical, width, height, d_gaussian_deriv, k_width, d_vertical_gradient);	

		// Horizontal Gradient
		convolve_ver<<<dmGrid,dmBlock,sizeof(float)*16*(16+2*floor(k_width/2))>>>
				(d_org_img, width, height, d_gaussian_kernel, k_width, d_temp_horizontal);	
		convolve_hor<<<dmGrid,dmBlock,sizeof(float)*16*(16+2*floor(k_width/2))>>>
				(d_temp_horizontal, width, height, d_gaussian_deriv, k_width, d_horizontal_gradient);	
		
		horizontal_gradient = (float*)malloc(sizeof(float)*height*width);
		vertical_gradient = (float*)malloc(sizeof(float)*height*width);
		temp_gradient = (float*)malloc(sizeof(float)*height*width);	
		temp_vertical = (float*)malloc(sizeof(float)*height*width);	

		cudaMemcpy(horizontal_gradient, d_horizontal_gradient, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
		cudaMemcpy(vertical_gradient, d_vertical_gradient, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
		
		gettimeofday(&end, NULL);

		write_image_template("h_gradient_L2.pgm", horizontal_gradient, width, height);
		write_image_template("v_gradient_L2.pgm", vertical_gradient, width, height);

		printf("%ld\n", (end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
	
		// Cuda Free
		cudaFree(d_org_img);
		cudaFree(d_temp_horizontal);
		cudaFree(d_temp_vertical);
		cudaFree(d_horizontal_gradient);
		cudaFree(d_vertical_gradient);
		cudaFree(d_gaussian_kernel);
		cudaFree(d_gaussian_deriv);
	}
}
