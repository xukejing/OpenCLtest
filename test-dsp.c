#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>
#define MICRO_IN_SEC 1000000.00
double microtime() {

	struct timeval tv;
	//struct timezone tz;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / MICRO_IN_SEC;
}

int get_ocl_string(const char *file_name, char *ocl_string)
{
    FILE *fp;
    int file_length;
    int status = 0;
    
    fp = fopen(file_name, "r");
    if (fp == NULL)
        return -1;
    
    fseek(fp, 0, SEEK_END);
    file_length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    status = fread(ocl_string, 1, file_length, fp);
    if (status == -1)
        return -1;

    return file_length;
}

int main(void)
{
    int array_a[20] = {0, 1, 8, 7, 6, 2, 3, 5, 4, 9,17,19,15,10,18,16,14,13,12,11};
    int array_b[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0};
    size_t datasize = 20 * sizeof(int);
    size_t ocl_string_size;
    char *ocl_string;
	double start_time, dt, dt_err;
	start_time = microtime();
	dt_err = microtime() - start_time;

    ocl_string = (char *)malloc(400);
	//ocl_string = (char *)malloc(20);

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_mem buffer_a, buffer_b;
    cl_program program;
    cl_kernel kernel;
	cl_event kernelEvent;
    
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
    
    context = clCreateContext(NULL, 1 ,&device_id, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device_id, 0, NULL);
    
    buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
    buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);

    
    ocl_string_size = get_ocl_string("test.cl", ocl_string);
    clEnqueueWriteBuffer(command_queue, buffer_a, CL_FALSE, 0, \
							datasize, array_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, buffer_b, CL_FALSE, 0, \
							datasize, array_b, 0, NULL, NULL);
    program = clCreateProgramWithSource(context, 1, (const char **)&ocl_string, \
							&ocl_string_size, NULL);

    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "test", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem),&buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem),&buffer_b);
 

    size_t global_work_size[1] = {20};
	start_time = microtime();
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, \
							global_work_size, NULL, 0, NULL, &kernelEvent);
	clWaitForEvents(1, &kernelEvent);
	clEnqueueReadBuffer(command_queue, buffer_b, CL_TRUE, 0, \
		datasize, array_b, 0, NULL, NULL);
	dt = microtime() - start_time - dt_err;
	printf("time = %lf s\n", dt);

    for (int i = 0 ; i < 20; i ++) {
        printf("%d ", array_a[i]);
    }
    printf("\n");
    for (int i = 0 ; i < 20; i ++) {
        printf("%d ", array_b[i]);
    }
    printf("\n");


    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseContext(context);

    return 0;
}
