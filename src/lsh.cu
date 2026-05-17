#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <chrono>
#include <system_error>
#include "mpi.h"
#include "param.h"
#include "util.h"
#include "lsh.h"
#include <thread>

// Macro for checking CUDA errors.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace fs = std::filesystem;
//Parameters
static int num_hash = 128;          // Number of hash functions
static int len_shingle = 5;         // Length of each shingle
static int b = 16;                   // Number of bands
static int num_file;                // Number of files
static int th;                      // Similarity threshold
static int max_bucket, num_key, buf_c;  // Bucket and key-related variables
static int file_offload;

// Hashing and CUDA memory related variables
static unsigned int *p, *q, *r;
static unsigned int *_p, *_q, *_r;
static int *bias_cuda;
static char *buf_cuda;
static unsigned int *hash_result, *total_hash_result;
static unsigned int *hash_result_cuda;
static unsigned int *buf[2][C], *cuda_buf1;
static int *file_idx[2][C], *line_idx[2][C];
static int cnt[2][C] = {0,}; // Initialize counters for each chunk
static int wr = 0;
static unsigned char* cuda_cmp_result;
static int* cuda_reduce_result;
static int* cuda_reduce_cnt;
static int* cuda_reduce_buf;
static unsigned char* cmp_result;
static int* reduce_result;
static int *par, *par_new;

#define REDUCE_BLOCK 240
#define REDUCE_EDGE_TH 50

// Find root of a node for union-find
int root(int x) {
    if(par[x]==x) return x;
    int tmp=root(par[x]);
    par[x]=tmp;
    return tmp;
}

// Merge two sets in union-find
void merge(int x, int y) {
    x=root(x);
    y=root(y);
    if(x==y) return;
    par[y]=x;
}

// Initialize LSH-related CUDA structures
void init_lsh_cuda(int _num_hash, int _len_shingle, int _b, int seed, double _th, int _num_file) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes
    num_hash=_num_hash;
    len_shingle=_len_shingle;
    b=_b;
    num_file=_num_file;
    th=num_hash*_th;
    srand(seed);
    p=(unsigned int*)malloc(sizeof(unsigned int)*num_hash);
    q=(unsigned int*)malloc(sizeof(unsigned int)*num_hash);
    r=(unsigned int*)malloc(sizeof(unsigned int)*num_hash);

    for(int i=0; i<num_hash; i++) {
        q[i]=4294967; //=prime & 4294967*1000 < 2^32-1
        p[i]=257+i;

        r[i]=q[i]-1;
        for(int j=0; j<len_shingle; j++) {
            r[i] = (r[i]*p[i])%q[i];
        }
    }

    gpuErrchk(cudaMalloc((void**)&_p, sizeof(unsigned int) * num_hash));
    gpuErrchk(cudaMalloc((void**)&_q, sizeof(unsigned int) * num_hash));
    gpuErrchk(cudaMalloc((void**)&_r, sizeof(unsigned int) * num_hash));

    gpuErrchk(cudaMemcpy(_p, p, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_q, q, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_r, r, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&hash_result_cuda, sizeof(unsigned int)*MAX_LINE*(num_hash+b)));
    gpuErrchk(cudaMalloc(&bias_cuda, sizeof(int)*(MAX_LINE+1)));
    gpuErrchk(cudaMalloc(&buf_cuda, 1e9));

    MPI_Barrier(MPI_COMM_WORLD);
    set_param(num_file,num_hash+b);
    get_param(num_key, max_bucket, buf_c, file_offload);
    MPI_Barrier(MPI_COMM_WORLD);

    hash_result= (unsigned int*)malloc(sizeof(unsigned int) * MAX_LINE * (num_hash+b));
    if(!file_offload) {
        total_hash_result= (unsigned int*)malloc(sizeof(unsigned int) * MAX_LINE * (num_hash+b) * num_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0; i<buf_c; i++) {
        buf[0][i]=(unsigned int*)malloc(sizeof(unsigned int) * num_hash * max_bucket);
        buf[1][i]=(unsigned int*)malloc(sizeof(unsigned int) * num_hash * max_bucket);
        file_idx[0][i]=(int*)malloc(sizeof(int) * max_bucket);
        file_idx[1][i]=(int*)malloc(sizeof(int) * max_bucket);
        line_idx[0][i]=(int*)malloc(sizeof(int) * max_bucket);
        line_idx[1][i]=(int*)malloc(sizeof(int) * max_bucket);
    }

    gpuErrchk(cudaMalloc((void**)&cuda_cmp_result, sizeof(unsigned char) * max_bucket * max_bucket));
    MPI_Barrier(MPI_COMM_WORLD);
    gpuErrchk(cudaMalloc((void**)&cuda_buf1, sizeof(unsigned int) * num_hash * max_bucket));

    gpuErrchk(cudaMalloc((void**)&cuda_reduce_result, sizeof(int) * max_bucket * REDUCE_EDGE_TH * 2)); 
    gpuErrchk(cudaMalloc((void**)&cuda_reduce_buf, sizeof(int) * REDUCE_BLOCK));
    gpuErrchk(cudaMalloc((void**)&cuda_reduce_cnt, sizeof(int)));

    cmp_result=(unsigned char*)malloc(sizeof(unsigned char) * max_bucket * max_bucket);
    reduce_result=(int*)malloc(sizeof(int) * max_bucket * max_bucket);

    par=(int*)malloc(sizeof(int)*MAX_LINE*num_file);
    par_new=(int*)malloc(sizeof(int)*MAX_LINE*num_file);

    
}

static double cmp_time1=0, cmp_time2=0, cmp_time3=0, cmp_time4=0, cmp_time5=0;
static double read_time = 0;     
static double buffer_time = 0;  

void print_cmp_time_lsh() {

    std::cout << "  - file read: " <<  read_time << " seconds" << std::endl;
    std::cout << "  - buffering: " <<  buffer_time << " seconds" << std::endl;
    
    std::cout << "  - file read + buffering: " <<  cmp_time1 << " seconds" << std::endl;
    std::cout << "  - Comm time1: " <<  cmp_time2 << " seconds" << std::endl;
    std::cout << "  - GPU kernel : " <<  cmp_time3 << " seconds" << std::endl;
    std::cout << "  - Comm time2: " <<  cmp_time4 << " seconds" << std::endl;
    std::cout << "  - Union time: " <<  cmp_time5 << " seconds" << std::endl;
}

// Kernel to compute hash values for strings
__global__ void hash_string_kernel_lsh(char *buf, int *bias, unsigned int *_p, unsigned int *_q, unsigned int *_r, int num_line, int len_shingle, int num_hash, int b, unsigned int *hash_result) {
    int line_id = blockIdx.x;       // Line index
    int hash_id = threadIdx.x;      // Hash index

    unsigned int sum = 0;           // Intermediate hash sum
    unsigned int res = 0;           // Minimum hash value

    int len = bias[line_id + 1] - bias[line_id] -1; // Length of the string segment
    if (len < len_shingle) return;              // Skip if segment length is less than shingle length

    char *text = buf + bias[line_id];          // Start of the text for this line

    unsigned int p = _p[hash_id];              // Prime coefficient for hash function
    unsigned int q = _q[hash_id];              // Modulus for hash function
    unsigned int r = _r[hash_id];              // Precomputed power for hash rolling

    // Compute hash for the initial window of length `len_shingle`
    // When the data only contains a 'text' field, the parsing process can be skipped and execution can be optimized by starting from the 10th character.
    for (int i = 10; i < 10 + len_shingle; i++) {
        sum = (sum * p + text[i]) % q;
    }
    res = sum;
    // Compute hash for the rolling window
    for (int i = 10 + len_shingle; i < len - len_shingle - 1 ; i++) {
        sum = (sum * p + ((unsigned int)text[i - len_shingle]) * r + text[i]) % q;
        res = min(res, sum);
    }
    // Store the resulting hash value
    hash_result[line_id * (num_hash + b) + hash_id] = res;
}



// Kernel to generate keys from hashed values
__global__ void generate_key_kernel(unsigned int *hash_result, int num_line, int num_hash, int b, int num_key) {
    int line_id = blockIdx.x;       // Line index
    int b_id = threadIdx.x;         // Band index

    unsigned int sum = 0;           // Sum of hash values for the band
    int h = num_hash / b;           // Number of hash functions per band

    // Sum hash values for the current band
    for(int i=b_id*h; i<(b_id+1)*h; i++) sum+=hash_result[line_id*(num_hash+b) + i];
    
    // Generate the band key
    hash_result[line_id*(num_hash+b) + num_hash+b_id] = sum%num_key;
}

static double total_computation_time, cpu_to_gpu, gpu_to_cpu, file_write_time;

double get_total_computation_time_lsh() {return total_computation_time;}
double c2g() {return cpu_to_gpu;}
double g2c() {return gpu_to_cpu;}
double get_total_file_write_time_lsh() {return file_write_time;}



// Function to perform LSH (Locality Sensitive Hashing) and write results to a binary file
void lsh_cuda_async(const std::string& filepath, const std::string& outputPath, char* buf, int* bias, int num_line,
                    int file_size, int file_idx, int num_file,
                    cudaStream_t stream){

    // CUDA events for timing
    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    // ===============================
    // Host -> Device copy timing
    // ===============================
    cudaEventRecord(startH2D, stream);

    gpuErrchk(cudaMemcpyAsync(bias_cuda, bias, sizeof(int)*(num_line+1),
                    cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(buf_cuda, buf, sizeof(char)*bias[num_line],
                    cudaMemcpyHostToDevice, stream));
    
    cudaEventRecord(stopH2D, stream);

    int blockSize = num_hash;
    int numBlocks = num_line;

    gpuErrchk(cudaGetLastError());

    cudaEventRecord(startKernel, stream);
     // Execute hash kernel on GPU
    hash_string_kernel_lsh<<<numBlocks, blockSize , 0 , stream>>>(buf_cuda, bias_cuda, _p, _q, _r, num_line, len_shingle, num_hash, b, hash_result_cuda);
    gpuErrchk(cudaGetLastError());
    // Execute key generation kernel on GPU
    generate_key_kernel<<<numBlocks, b , 0 , stream>>>(hash_result_cuda, num_line, num_hash, b, num_key);
    gpuErrchk(cudaGetLastError());
    cudaEventRecord(stopKernel, stream);

    if(file_offload) {
        // Copy results back to host memory
        cudaEventRecord(startD2H, stream);
        gpuErrchk(cudaMemcpy(hash_result, hash_result_cuda, sizeof(unsigned int)*num_line*(num_hash+b), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaGetLastError());
        cudaEventRecord(stopD2H, stream);
        cudaEventSynchronize(stopD2H);  // wait D2H done

        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        auto time_write1 = std::chrono::high_resolution_clock::now();
        // Construct output file path
        std::filesystem::path inputPath(filepath);
        std::string filename = inputPath.stem().string();
        std::string newFilename = filename + "_hashresult.bin";
        
        std::filesystem::path outputDir(outputPath);
        std::filesystem::path outputFilePath = outputDir / newFilename;
        
        // Write hash results to binary file
        std::ofstream outFile(outputFilePath, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << outputFilePath << std::endl;
            return;
        }
        outFile.write(reinterpret_cast<const char*>(hash_result), sizeof(unsigned int)*num_line*(num_hash+b));
        outFile.close();
        auto time_write2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed1 = time_write2-time_write1;
        file_write_time+=elapsed1.count();
    } else {
        cudaEventRecord(startD2H, stream);
        gpuErrchk(cudaMemcpy(total_hash_result+MAX_LINE*(num_hash+b)*file_idx, hash_result_cuda, sizeof(unsigned int)*num_line*(num_hash+b), cudaMemcpyDeviceToHost));
        file_write_time =0;
        cudaEventRecord(stopD2H, stream);
        cudaEventSynchronize(stopD2H);
    }
    // ===============================
    // Measure elapsed times
    // ===============================
    float msH2D = 0, msKernel = 0, msD2H = 0;
    cudaEventSynchronize(stopH2D);
    cudaEventSynchronize(stopKernel);

    cudaEventElapsedTime(&msH2D, startH2D, stopH2D);
    cudaEventElapsedTime(&msKernel, startKernel, stopKernel);
    cudaEventElapsedTime(&msD2H, startD2H, stopD2H);

    cpu_to_gpu += (msH2D / 1000.0);            
    total_computation_time += (msKernel / 1000.0);
    gpu_to_cpu += (msD2H / 1000.0);

    // ===============================
    // Cleanup events
    // ===============================
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);
    
}

// Gathers the file sizes from all processes into the file_size array
void AllgatherFileSize(int size, int files_per_process, int *file_size) {

    // Allocate memory for the send counts array to store the number of files per process
    int *sendcounts = (int*)malloc(sizeof(int) * size);
    
    // Gather the number of files each process has
    MPI_Allgather(&files_per_process, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Allocate memory for the displacement array to store the starting index of each process's data
    int *displs = (int*)malloc(sizeof(int) * size);
    displs[0] = 0;
    // Calculate the displacement values based on the send counts
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
    // Gather file sizes from all processes into the file_size array
    MPI_Allgatherv(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,  
        file_size, sendcounts, displs, MPI_INT,
        MPI_COMM_WORLD
    );
    free(sendcounts);
    free(displs);
    }

// Gathers the hash results from all processes into the total_hash_result array
void AllgatherHashResult(int rank, int size, int files_per_process, int start_index) {
    if(file_offload) return;
    // Allocate memory for the send counts and displacement arrays
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
   
    // Gather the number of files each process handles
    MPI_Allgather(&files_per_process, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate the data size each process will send (including hash data)
    for (int i = 0; i < size; i++) {
        sendcounts[i] = MAX_LINE * sendcounts[i] * (num_hash + b);
    }

    // Calculate the displacement for each process's data in the result buffer
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    // Gather the hash results from all processes into the total_hash_result array
    MPI_Allgatherv(
        total_hash_result + MAX_LINE * start_index * (num_hash + b),  
        sendcounts[rank],                                            
        MPI_INT,                                                     
        total_hash_result,                                           
        sendcounts,                                                 
        displs,                                                     
        MPI_INT,                                                    
        MPI_COMM_WORLD                                              
    );

    // Free dynamically allocated memory
    free(sendcounts);
    free(displs);
    MPI_Barrier(MPI_COMM_WORLD);  
    
}

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define THREAD_NUM BLOCK_SIZE*BLOCK_SIZE

// Kernel function to compare hash values and determine similarity
__global__ void compare_lsh_kernel(unsigned int *buf, unsigned char* result, int line_num, int num_hash, int th) {
    // Inputs:
    // - buf: Pointer to the (Minhash signatures+bucket id).
    // - result: Pointer to the output matrix, storing pairwise comparison results.
    // - line_num: Total number of hash signatures.
    // - num_hash: Number of hash values per signature.
    // - th: Threshold for similarity check.     // ex) hash num =128, th = 0.8 => th = 128*0.8 = 102.4

    int block_id1 = blockIdx.x;
    int block_id2 = blockIdx.y;
    if(block_id2 < block_id1) return; 

    // Declare shared memory for storing tiles of hash values for two blocks.
    __shared__ int t1[BLOCK_SIZE*TILE_SIZE];
    __shared__ int t2[BLOCK_SIZE*TILE_SIZE];

    // Calculate the range of rows handled by block_id1 and block_id2.
    int x=threadIdx.x;
    int dx = x%BLOCK_SIZE;
    int dy = x/BLOCK_SIZE;

    int bias1=(BLOCK_SIZE*block_id1), lim1 = min(BLOCK_SIZE*(block_id1+1), line_num);
    int bias2=(BLOCK_SIZE*block_id2), lim2 = min(BLOCK_SIZE*(block_id2+1), line_num);
    
    // Get pointers to the start of each block's data in the global memory.
    unsigned int *buf1 = &buf[(long long)bias1 * num_hash];
    unsigned int *buf2 = &buf[(long long)bias2 * num_hash];

    unsigned int cnt=0;

    // Loop through the hash values in tiles of size TILE_SIZE.
    for(int tile=0; tile<num_hash; tile+=TILE_SIZE) {
        // Load TILE_SIZE hash values into shared memory for each block.
        if(bias1+dy<lim1) t1[dy*TILE_SIZE+dx] = buf1[dy * num_hash + tile + dx];
        if(bias2+dy<lim2) t2[dy*TILE_SIZE+dx] = buf2[dy * num_hash + tile + dx];
        __syncthreads();

        // Compare hash values in shared memory to count matches.
        for(int k = 0; k < TILE_SIZE; k += 4) {
            cnt += (t1[dx*TILE_SIZE + k] == t2[dy*TILE_SIZE + k]);
            cnt += (t1[dx*TILE_SIZE + k + 1] == t2[dy*TILE_SIZE + k + 1]);
            cnt += (t1[dx*TILE_SIZE + k + 2] == t2[dy*TILE_SIZE + k + 2]);
            cnt += (t1[dx*TILE_SIZE + k + 3] == t2[dy*TILE_SIZE + k + 3]);
        }

        __syncthreads();
    }
    // Check if the count is greater than threshold .
    if((bias1+dx)<lim1 && (bias2+dy) <lim2 && (bias1+dx)<(bias2+dy)) result[(size_t)(bias1+dx)*line_num+(bias2+dy)]=cnt>th;
}

#define REDUCE_THREAD 32
// The `reduce_compare_result` kernels are responsible for processing and compressing the necessary information
// from the comparison result matrix (`result`) before transferring it to the CPU.

// Kernel to reduce the comparison results and count the total number of set bits
__global__ void reduce_compare_result1(unsigned char* result, int *cuda_reduce_buf, int line_num) {
    __shared__ int prefix_sum[REDUCE_THREAD]; // Shared memory for prefix sum within a block
    int x = threadIdx.x;                     // Thread index within the block
    int b = blockDim.x;                      // Total threads in the block
    int y = blockIdx.x;                      // Block index
    int cnt = 0;                             // Local counter for set bits

    // Compute the total number of elements (rounded down to the nearest multiple of 4)
    size_t total = (size_t)line_num * line_num / 4 * 4;

    // Define the range of indices to process for this block
    size_t l = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * y / 4 * 4;
    size_t r = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * (y + 1) / 4 * 4;
    if (r > total) r = total;

    unsigned int *result_m = (unsigned int *)result; // Treat result as an array of 32-bit integers

    // Count the number of set bits in the result within the assigned range
    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        cnt += __popc(val); // Count set bits in the 32-bit word using CUDA's __popc intrinsic
    }
    // Store the count in shared memory
    prefix_sum[x] = cnt;
    __syncthreads();

    // Perform a parallel prefix sum within the block
    for (int k = 1; k < REDUCE_THREAD; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    // Store the total count for this block in the global buffer
    if (x == 0) cuda_reduce_buf[y] = prefix_sum[REDUCE_THREAD - 1];
}

// Kernel to compute the total count of set bits across all blocks
__global__ void reduce_compare_result2(int* cuda_reduce_buf, int *total) {
    __shared__ int prefix_sum[REDUCE_BLOCK]; // Shared memory for prefix sum within the block
    int x = threadIdx.x;                     // Thread index within the block

    // Load the counts from the global buffer into shared memory
    prefix_sum[x] = cuda_reduce_buf[x];
    __syncthreads();

    // Perform a parallel prefix sum within the block
    for (int k = 1; k < REDUCE_BLOCK; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    // Store the total count in the output variable and update the global buffer
    if (x == 0) {
        *total = prefix_sum[REDUCE_BLOCK - 1];
    }
    cuda_reduce_buf[x] = prefix_sum[x];
}

// Kernel to extract detailed comparison results and store matched pairs in the output
__global__ void reduce_compare_result3(unsigned char* result, int *cuda_reduce_buf, int *output, int line_num) {
    __shared__ int prefix_sum[REDUCE_THREAD]; // Shared memory for prefix sum within the block
    int x = threadIdx.x;                     // Thread index within the block
    int b = blockDim.x;                      // Total threads in the block
    int y = blockIdx.x;                      // Block index
    int cnt = 0;                             // Local counter for set bits

    // Compute the total number of elements (rounded down to the nearest multiple of 4)
    size_t total = (size_t)line_num * line_num / 4 * 4;

    // Define the range of indices to process for this block
    size_t l = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * y / 4 * 4;
    size_t r = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * (y + 1) / 4 * 4;
    if (r > total) r = total;

    unsigned int *result_m = (unsigned int *)result; // Treat result as an array of 32-bit integers

    // Count the number of set bits in the result within the assigned range
    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        cnt += __popc(val); // Count set bits in the 32-bit word
    }

    // Store the count in shared memory
    prefix_sum[x] = cnt;
    __syncthreads();

    // Perform a parallel prefix sum within the block
    for (int k = 1; k < REDUCE_THREAD; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    // Adjust the count using the global buffer
    cnt = prefix_sum[x] + cuda_reduce_buf[y] - prefix_sum[REDUCE_THREAD - 1];

    // Iterate through the result and extract matching pairs
    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        
        // Extract individual matches from the 32-bit word
        if (val & 1) {
            cnt--;
            output[cnt * 2] = (int)(i % line_num);     // Row index
            output[cnt * 2 + 1] = (int)(i / line_num); // Column index
        }
        if (val & 0x100) {
            cnt--;
            output[cnt * 2] = (int)((i + 1) % line_num);
            output[cnt * 2 + 1] = (int)((i + 1) / line_num);
        }
        if (val & 0x10000) {
            cnt--;
            output[cnt * 2] = (int)((i + 2) % line_num);
            output[cnt * 2 + 1] = (int)((i + 2) / line_num);
        }
        if (val & 0x1000000) {
            cnt--;
            output[cnt * 2] = (int)((i + 3) % line_num);
            output[cnt * 2 + 1] = (int)((i + 3) / line_num);
        }
    }
}

void compare_lsh_cuda(const vector<string> &file_list, const std::string& outputPath, int num_file, int *file_size, int rank, int size) {
    std::thread p;
    bool p_alive = false;

    // Initialize the parent array for union-find (disjoint-set) operations
    for(int i=0; i<num_file*MAX_LINE; i++) par[i]=i;

    // Record the start time of the function
    int b_start, b_end, b_step;       
    int key_start, key_end, key_step; 
    if (b >= size) {
        // b ≥ rank → round-robin
        b_start = rank;
        b_end   = b;
        b_step  = size;

        key_start = 0;
        key_end   = num_key;
        key_step  = C;

    } else {
        // b < rank
        int procs_per_band = size / b;
        int my_band        = rank / procs_per_band;
        int my_subrank     = rank % procs_per_band;

        b_start = my_band;
        b_end   = my_band + 1;   
        b_step  = 1;

        int keys_per_proc = (num_key + procs_per_band - 1) / procs_per_band;
        key_start = my_subrank * keys_per_proc;
        key_end   = std::min(num_key, key_start + keys_per_proc);
        key_step  = C;
    }
    
    // Record the start time of the function
    auto time_1 = std::chrono::high_resolution_clock::now();

    for(int b_id = b_start; b_id < b_end; b_id += b_step) {
      for(int key = key_start; key < key_end; key += key_step) {
            auto time1 = std::chrono::high_resolution_clock::now();
            for (int c = 0; c < C; ++c) cnt[wr][c] = 0;

            // Loop through each file in the file list
            for(int i=0; i<num_file; i++) {
                int fs = file_size[i];                // The number of documents in the file

                auto t_read_start = std::chrono::high_resolution_clock::now();
                if(file_offload) {
                    std::string filepath = file_list[i];   // Get the file path

                    // Prepare the path to the hash result binary file
                    std::filesystem::path inputPath(filepath);
                    std::string filename = inputPath.stem().string();
                    std::string newFilename = filename + "_hashresult.bin";
                    std::filesystem::path outputDir(outputPath);
                    std::filesystem::path outputFilePath = outputDir / newFilename;

                    // Open the hash result file for reading
                    std::ifstream inFile(outputFilePath, std::ios::binary);
                    if (!inFile) {
                        std::cerr << "Error opening file for reading: " << outputFilePath << std::endl;
                        return;
                    }
                    // Read the hash results into memory
                    inFile.read(reinterpret_cast<char*>(hash_result), sizeof(unsigned int) * fs * (num_hash+b));
                    inFile.close();
                } else {
                    memcpy( 
                        hash_result, 
                        total_hash_result + MAX_LINE * (num_hash + b) * i, 
                        sizeof(unsigned int) * fs * (num_hash + b)
                    );
                }
                auto t_read_end = std::chrono::high_resolution_clock::now();
                read_time += std::chrono::duration<double>(t_read_end - t_read_start).count();

                auto t_buf_segment_start = std::chrono::high_resolution_clock::now();
                // Organize hash results into buckets based on the key
                for(int j=0; j<fs; j++) {
                    auto tmp = *(hash_result + (unsigned long long)j*(num_hash+b) + num_hash + b_id) - key;
                    if(tmp < C) {
                        // Copy hash results for the current bucket into the buffer
                        memcpy(buf[wr][tmp]+cnt[wr][tmp]*num_hash, hash_result+(unsigned long long)j*(num_hash+b), sizeof(unsigned int)*num_hash);
                        file_idx[wr][tmp][cnt[wr][tmp]] = i;
                        line_idx[wr][tmp][cnt[wr][tmp]] = j;
                        cnt[wr][tmp]++;

                        // If the number of documents in a bucket exceeds max_bucket,  
                        // perform pairwise comparisons in chunks of max_bucket within that bucket.
                        if(cnt[wr][tmp]==max_bucket) {
                            if (p_alive) { p.join(); p_alive = false; }
                            // printf("  bucket reset: %d %d %d\n", key, i, j);
                            int c=tmp;
                            gpuErrchk(cudaMemcpy(cuda_buf1, buf[wr][c], sizeof(unsigned int)*cnt[wr][c]*num_hash, cudaMemcpyHostToDevice));

                            const int block = BLOCK_SIZE;
                            dim3 numBlocks((cnt[wr][c] + block - 1) / block, (cnt[wr][c] + block - 1) / block);
                            dim3 blockSize(THREAD_NUM);
                            gpuErrchk(cudaMemset(cuda_cmp_result, 0, sizeof(unsigned char) * cnt[wr][c] * cnt[wr][c]));

                            compare_lsh_kernel<<<numBlocks, blockSize>>>(cuda_buf1, cuda_cmp_result, cnt[wr][c], num_hash, th);

                            cudaDeviceSynchronize();
                            gpuErrchk(cudaGetLastError());
                            auto time5 = std::chrono::high_resolution_clock::now();

                            gpuErrchk(cudaMemcpy(cmp_result, cuda_cmp_result, sizeof(unsigned char)*cnt[wr][c]*cnt[wr][c], cudaMemcpyDeviceToHost));

                            for(int i=0; i<cnt[wr][c]; i++) {
                                for(int j=0; j<cnt[wr][c]; j++) {
                                    if(i==j) continue;
                                    if(cmp_result[i*cnt[wr][c]+j] > th) {
                                        int x = file_idx[wr][c][i]*MAX_LINE + line_idx[wr][c][i];
                                        int y = file_idx[wr][c][j]*MAX_LINE + line_idx[wr][c][j];
                                        merge(x, y); // Merge related data
                                    }
                                }
                            }
                            cnt[wr][c] = 0;
                        }
                    }
                }
                auto t_buf_segment_end = std::chrono::high_resolution_clock::now();
                buffer_time += std::chrono::duration<double>(t_buf_segment_end - t_buf_segment_start).count();
            
            }

            auto time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = time2-time1;
            cmp_time1 += elapsed1.count();

            // Synchronize CUDA device and check for errors

            if (p_alive) { p.join(); p_alive = false; }
            int prev = wr;
            wr ^= 1;

            // Process each chunk in the range of C
            p = std::thread([=]() {
                int rank = -1;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                int deviceCount = 0;
                cudaGetDeviceCount(&deviceCount);

                int dev = rank % deviceCount;
                cudaSetDevice(dev);
                for (int c = 0; c < C; ++c) {
                    int n = cnt[prev][c];
                    if (n <= 0) continue;
                    if (n > max_bucket) {
                        continue;
                    }

                    // H2D
                    auto time2 = std::chrono::high_resolution_clock::now();
                    gpuErrchk(cudaMemcpy(cuda_buf1,
                                        buf[prev][c],
                                        sizeof(unsigned int) * (size_t)n * num_hash,
                                        cudaMemcpyHostToDevice));
                    auto time3 = std::chrono::high_resolution_clock::now();
                    // compare kernel
                    const int block = BLOCK_SIZE;
                    dim3 numBlocks((n + block - 1) / block, (n + block - 1) / block);
                    dim3 blockSize(THREAD_NUM);
                    gpuErrchk(cudaMemset(cuda_cmp_result, 0, sizeof(unsigned char) * (size_t)n * n));
                    compare_lsh_kernel<<<numBlocks, blockSize>>>(cuda_buf1, cuda_cmp_result, n, num_hash, th);
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaGetLastError());
                    auto time4 = std::chrono::high_resolution_clock::now();
                    // reduce
                    reduce_compare_result1<<<REDUCE_BLOCK, REDUCE_THREAD>>>(cuda_cmp_result, cuda_reduce_buf, n);
                    reduce_compare_result2<<<1, REDUCE_BLOCK>>>(cuda_reduce_buf, cuda_reduce_cnt);
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaGetLastError());
                    auto time5 = std::chrono::high_resolution_clock::now();
                    int reduce_cnt = 0;
                    gpuErrchk(cudaMemcpy(&reduce_cnt, cuda_reduce_cnt, sizeof(int), cudaMemcpyDeviceToHost));

                    if (reduce_cnt > max_bucket * REDUCE_EDGE_TH) {
                        gpuErrchk(cudaMemcpy(cmp_result,
                                            cuda_cmp_result,
                                            sizeof(unsigned char) * (size_t)n * n,
                                            cudaMemcpyDeviceToHost));
                        for (int i = 0; i < n; ++i) {
                            for (int j = 0; j < n; ++j) {
                                if (i == j) continue;
                                if (cmp_result[i * n + j] > th) {
                                    int x = file_idx[prev][c][i] * MAX_LINE + line_idx[prev][c][i];
                                    int y = file_idx[prev][c][j] * MAX_LINE + line_idx[prev][c][j];
                                    merge(x, y);
                                }
                            }
                        }
                    } else if (reduce_cnt > 0) {
                        reduce_compare_result3<<<REDUCE_BLOCK, REDUCE_THREAD>>>(
                            cuda_cmp_result, cuda_reduce_buf, cuda_reduce_result, n);
                        gpuErrchk(cudaMemcpy(reduce_result,
                                            cuda_reduce_result,
                                            sizeof(int) * reduce_cnt * 2,
                                            cudaMemcpyDeviceToHost));
                        for (int id = 0; id < reduce_cnt; ++id) {
                            int i = reduce_result[id + id];
                            int j = reduce_result[id + id + 1];
                            if (i == j) continue;
                            int x = file_idx[prev][c][i] * MAX_LINE + line_idx[prev][c][i];
                            int y = file_idx[prev][c][j] * MAX_LINE + line_idx[prev][c][j];
                            merge(x, y);
                        }
                    }
                auto time6 = std::chrono::high_resolution_clock::now();
                // Record elapsed times for different phases
                std::chrono::duration<double> elapsed2 = time3-time2;
                std::chrono::duration<double> elapsed3 = time4-time3;
                std::chrono::duration<double> elapsed4 = time5-time4;
                std::chrono::duration<double> elapsed5 = time6-time5;
                cmp_time2 += elapsed2.count();
                cmp_time3 += elapsed3.count();
                cmp_time4 += elapsed4.count();
                cmp_time5 += elapsed5.count();
                }
            });

            p_alive = true;

            auto time_2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_t = time_2-time_1;

            if(rank == 0) {
                std::cout << "rank:" << rank << ", b_id:" << b_id << "/" << b  << ", key:" << key << "/" << num_key << ", time: " << elapsed_t.count() << "\n";
                print_cmp_time_lsh();
            }
        }
    }
    if (p_alive) p.join();
}


void merge_union(int rank, int size) {
    // Perform a tree-based merge operation across ranks
    for(int r=1; r<size; r=r+r) {
        if(rank%r) break; 
        if((rank/r)&1) {
            if (rank - r >= 0){
            // If the rank is an odd member of the group, send its data to the even member
            MPI_Send(par, MAX_LINE*num_file, MPI_INT, rank-r, 0, MPI_COMM_WORLD);
            }
        } else {
            if (rank + r < size) {
            // If the rank is an even member of the group, receive data from the odd member
            MPI_Recv(par_new, MAX_LINE*num_file, MPI_INT, rank+r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Merge the received data into the current rank's data
            for(int i=0; i<MAX_LINE*num_file; i++){
                if(par_new[i]==i) {
                    continue; // Skip if the parent is itself
                }
                merge(par_new[i], i); // Merge the parent-child relationship
            }
            }
        }
    }
    
    // Broadcast the final parent array from rank 0 to all other ranks
    MPI_Bcast(par, MAX_LINE*num_file, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int *group_size = (int*)malloc(MAX_LINE * num_file * sizeof(int));
        if (group_size == NULL) {
            perror("Memory allocation failed");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        for (int i = 0; i < MAX_LINE * num_file; i++) {
            group_size[i] = 0; 
        }
    
        // Calculate group sizes
        for (int i = 0; i < MAX_LINE * num_file; i++) {
            int root_i = root(i);
            group_size[root_i]++;
        }
    
        // Calculate duplicate document count and group count
        int duplicate_count = 0;
        int group_count = 0;
        for (int i = 0; i < MAX_LINE * num_file; i++) {
            if (group_size[i] >= 2) {  // Only consider groups with size 2 or more
                duplicate_count += group_size[i];
                group_count++;
            }
        }
        printf("\n");
        printf("Number of documents that have near-duplicate pairs: %d\n", duplicate_count);
        printf("Number of duplicate groups: %d\n", group_count);
        printf("Number of documents to be removed: %d\n", duplicate_count - group_count);
        printf("\n");
        free(group_size);
    }
}

void generate_file_init(const std::string& outputPath) {
    // Create the output directory if it does not exist
    if (!fs::exists(outputPath)) {
        fs::create_directories(outputPath);
    }
}

void delete_hash_result(const std::string& outputPath) {
    if(!file_offload) return;
    std::filesystem::path outputDir(outputPath);
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            std::error_code ec;
            std::filesystem::remove(entry.path(), ec);
        }
    }
}

void generate_file(const std::string& filepath, int file_idx, const std::string& outputPath) {
    // Open the input file for reading
    std::ifstream inputFile(filepath);
    if (!inputFile.is_open()) {
        std::cerr << "Can't open the file: " << filepath << std::endl;
        return;
    }

    // Prepare the output file path based on the input filename
    std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
    std::string outputFilePath = outputPath + "/" + filename;
    std::ofstream outputFile(outputFilePath);
    std::string line;

    // Initialize indices for processing the file
    int cur = file_idx * MAX_LINE; // Start index for the current file
    int tt = 0, rm = 0;            // Total lines processed (tt) and lines removed (rm)
    while (std::getline(inputFile, line)) {
        if(par[cur] == cur) {
            // Write the line to the output file if it is the root of its set
            outputFile << line << "\n";
        } else {
            rm++; // Count the removed line
        }
        tt++; // Count the total lines processed
        cur++; // Move to the next line index
    }
    inputFile.close();
    outputFile.close();
}

void finalize_lsh() {
    // Synchronize the CUDA device to ensure all operations are completed
    cudaDeviceSynchronize();
    // Free allocated host memory
    free(p); free(q); free(r);
    for(int i=0; i<C; i++) {
        free(buf[0][i]);
        free(buf[1][i]);
        free(file_idx[0][i]);
        free(file_idx[1][i]);
        free(line_idx[0][i]);
        free(line_idx[1][i]);
    }
    // Free allocated GPU memory
    cudaFree(bias_cuda);
    cudaFree(buf_cuda);
    cudaFree(hash_result_cuda);
    cudaFree(cuda_buf1);
    cudaFree(cuda_cmp_result);
    free(cmp_result);
    free(par); free(par_new);
}