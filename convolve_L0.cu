#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "sys/time.h"
#include "cuda.h"
#include "image_template.h"

__device__
int inBounds(int x, int y, int h, int w){
	if(x < 0 || x > w)
		return 0;
	else if(y < 0 || y > h)
		return 0;
	else
		return 1;
}

__global__
void convolve(float *image, int width, int height, float *mask, int mask_width, int mask_height, 
		float* out_image){
	
	int i, j, k, m;
	float sum;
	int offseti, offsetj;

	i=blockIdx.x*blockDim.x + threadIdx.x;
	j=blockIdx.y*blockDim.y + threadIdx.y;


	if(i<height && j < width){
		sum=0;
		for(k=0; k<mask_height;k++){
			for(m=0; m<mask_width; m++){
				offseti = -1*mask_height/2+k;
				offsetj = -1*mask_width/2+m;
					
				if(inBounds(j+offsetj, i+offseti, height, width))
					sum+= *(image+(i+offseti)*width+j+offsetj)*(*(mask+(k*mask_width)+m));
			}
		}
		*(out_image+(i*width)+j)=sum;
	}
		
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
		float *vertical_gradient, *horizontal_gradient, *temp_gradient;

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

		convolve<<<dmGrid,dmBlock>>>(d_org_img, width, height, d_gaussian_kernel, 1, k_width, d_temp_horizontal);	
		convolve<<<dmGrid,dmBlock>>>(d_temp_horizontal, width, height, d_gaussian_deriv, k_width, 1, d_horizontal_gradient);	

		convolve<<<dmGrid,dmBlock>>>(d_org_img, width, height, d_gaussian_kernel, k_width, 1, d_temp_vertical);	
		convolve<<<dmGrid,dmBlock>>>(d_temp_vertical, width, height, d_gaussian_deriv, 1, k_width, d_vertical_gradient);	
		
		horizontal_gradient = (float*)malloc(sizeof(float)*height*width);
		vertical_gradient = (float*)malloc(sizeof(float)*height*width);
		temp_gradient = (float*)malloc(sizeof(float)*height*width);
		

		cudaMemcpy(horizontal_gradient, d_horizontal_gradient, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
		cudaMemcpy(vertical_gradient, d_vertical_gradient, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
		
		gettimeofday(&end, NULL);

		write_image_template("h_gradient_L0.pgm", horizontal_gradient, width, height);
		write_image_template("v_gradient_L0.pgm", vertical_gradient, width, height);

		printf("%ld\n", (end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));

		// cuda free
		cudaFree(d_org_img);
		cudaFree(d_temp_horizontal);
		cudaFree(d_temp_vertical);
		cudaFree(d_horizontal_gradient);
		cudaFree(d_vertical_gradient);
		cudaFree(d_gaussian_kernel);
		cudaFree(d_gaussian_deriv);	
	}
}
