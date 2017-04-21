#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "sys/time.h"
#include "image_template.h"

int inBounds(int x, int y, int h, int w){
	if(x < 0 || x >= w)
		return 0;
	else if(y < 0 || y >= h)
		return 0;
	else
		return 1;
}

void convolve(float *image, float **output, float *kernel, int height, 
		int width, int k_height, int k_width){
	
	*output = (float*)malloc(sizeof(float)*height*width);
	
	// Iter pixels
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			float sum = 0;
			// Iter kernel
			for(int k=0; k<k_height;k++){
				for(int m=0; m<k_width; m++){
					int offseti = -1*floor(k_height/2)+k;
					int offsetj = -1*floor(k_width/2)+m;
					
					if(inBounds(j+offsetj, i+offseti, height, width))
						sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
				}
			}
			*(*output+(i*width)+j)=sum;
		}
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
		float *g_kernel, *dg_kernel, *t_hor, *t_ver, *image, *h_grad, *v_grad;	

		read_image_template(argv[1],
				    &image,
				    &width,
				    &height);
		
		create_gaussians(&g_kernel, &dg_kernel, atof(argv[2]), &k_width);
	
		printf("Gaussian Kernel:\n");
		print_matrix(g_kernel, 1, k_width);
		printf("Derivative Kernel:\n");
		print_matrix(dg_kernel,1,k_width);

		gettimeofday(&start, NULL);
		// Horizonal gradient
		convolve(image, &t_hor, g_kernel, height, width, k_width, 1);
		convolve(t_hor, &h_grad, dg_kernel, height, width, 1, k_width);
		
		// Vertcal gradient
		convolve(image, &t_ver, g_kernel, height, width, 1, k_width);
		convolve(t_ver, &v_grad, dg_kernel, height, width, k_width, 1);
		gettimeofday(&end, NULL);

		write_image_template("h_gradient.pgm", h_grad, width, height);

		write_image_template("v_gradient.pgm", v_grad, width, height);

		printf("%ld\n", (end.tv_sec *1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
	}
}
