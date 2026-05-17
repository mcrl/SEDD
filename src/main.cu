#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#include "mpi.h"
#include "param.h"
#include "util.h"
#include "lsh.h"

#include <fstream>
#include <filesystem>
#include <string>

using namespace std;

// double buffer host pinned memory
char* h_buf[2] = {nullptr, nullptr};
int*  h_bias[2] = {nullptr, nullptr};

cudaStream_t stream[2];

#define MAX_LINE 30000

int filesize(const std::string& filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return (int)in.tellg(); 
}

void readFile(const std::string& filepath, int &num_line, int *&_bias, char* &_buf) {
    int total_size = filesize(filepath);

    std::ifstream file(filepath, std::ios::binary);
    file.read(_buf, total_size);
    file.close();

    int idx = 0;
    _bias[0] = 0;
    for(int i=0;i<total_size;i++){
        if(_buf[i]=='\n'){
            idx++;
            _bias[idx] = i+1;
        }
    }
    num_line = idx;
    _bias[num_line] = total_size;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    int deviceCount = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&deviceCount); // Get number of GPUs
    if (cuda_status != cudaSuccess) {return 1;}
    cudaSetDevice(rank % deviceCount); // Assign GPU to the process

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Directory path require" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string folderPath = argv[1]; // Input folder path
    std::string outputPath = argv[2]; // Output folder path

    vector<string> file_list;
    getFileList(file_list, folderPath); // Get list of files in the input folder
    sort(file_list.begin(), file_list.end()); // Sort files alphabetically

    int num_file = file_list.size(); // Total number of files

    if (rank == 0){
        if (num_file < size) {
            std::cerr << "Error: Number of files (" << num_file 
                      << ") is less than number of MPI processes (" << size 
                      << "). Please use fewer MPI processes." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
        
    int files_per_process = num_file / size;    
    int extra = num_file % size;               

    int start_index = rank * files_per_process + std::min(rank, extra);  // Start index for the process
    int end_index = start_index + files_per_process ; // End index for the process

    if (rank < extra) end_index++;      
    // Files assigned per process
    files_per_process = end_index - start_index ;

    // Initialize parameters for LSH
    int num_hash = NUM_HASH;
    int b = BUCKET;
    int shingle_len = SHINGLE_LEN;
    init_lsh_cuda(num_hash, shingle_len, b, 777984, 0.8, num_file); //num_hash, shingle_len, random_seed
    
    // Host pinned memory double buffer
    cudaMallocHost((void**)&h_buf[0], 1e9);   
    cudaMallocHost((void**)&h_buf[1], 1e9);
    cudaMallocHost((void**)&h_bias[0], sizeof(int)*(MAX_LINE+1));
    cudaMallocHost((void**)&h_bias[1], sizeof(int)*(MAX_LINE+1));

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    
    
    if(!rank) generate_file_init(outputPath);
    int *file_size= (int*)malloc(sizeof(int) * num_file);
    // Generate Minhash signature matrix and Calculate the bucket IDs of each band.
    if (rank == 0) {
        std::cout << "Start Minhash Generation.." << std::endl;
    }

    auto time1 = std::chrono::high_resolution_clock::now();

    int curr = 0;
    int num_line[2];

    static double total_file_read_time;
    
    for (int i = start_index; i < end_index; i++) {
        auto time_read1 = std::chrono::high_resolution_clock::now();
        int next = 1 - curr;
        readFile(file_list[i], num_line[next], h_bias[next], h_buf[next]);
        auto time_read2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> minhash_read = time_read2-time_read1;
        total_file_read_time += minhash_read.count();

        file_size[i] = num_line[next]; 
        
        if (i > start_index) {
            lsh_cuda_async(file_list[i-1], outputPath, h_buf[curr], h_bias[curr], num_line[curr],
                        file_size[i-1], i-1, num_file, stream[curr]);
        }
        curr = next;
    }
    lsh_cuda_async(file_list[end_index-1], outputPath, h_buf[curr], h_bias[curr], num_line[curr],
                   file_size[end_index-1], end_index-1, num_file, stream[curr]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);


    cudaFreeHost(h_buf[0]);
    cudaFreeHost(h_buf[1]);
    cudaFreeHost(h_bias[0]);
    cudaFreeHost(h_bias[1]);
    // Gather file sizes from all processes    
    AllgatherFileSize(size, files_per_process, file_size);

    auto time2 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed1 = time2 - time1;
    
    // Calculate time taken for MinHash
    if (rank == 0) {
        printf("==================================================\n\n");
        std::cout << "Min Hash total time: " << elapsed1.count() << " seconds" << std::endl;
        printf("\n");
        printf("==================================================\n\n");
        // std::cout << "  - File read time: " << total_file_read_time << " seconds" << std::endl;
        // std::cout << "  - Computation time(c2g): " << c2g() << " seconds" << std::endl;
        // std::cout << "  - Computation time: " << get_total_computation_time_lsh() << " seconds" << std::endl;
        // std::cout << "  - Computation time(g2c): " << g2c() << " seconds" << std::endl;
        // std::cout << "  - File write time: " << get_total_file_write_time_lsh() << " seconds\n" << std::endl;
    }

    // Comparison phase
    if (rank == 0) {
        std::cout << "Start Comparison.." << std::endl;
        printf("\n\n");
    }

    // (when file offloading is disabled) 
    // Gathers the hash results from all processes into the total_hash_result array
    AllgatherHashResult(rank, size, files_per_process, start_index);
    // cleanup_file_writer();
    time1 = std::chrono::high_resolution_clock::now();
    compare_lsh_cuda(file_list, outputPath, num_file, file_size, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = time2 - time1;
    if (rank == 0) {
        printf("==================================================\n");
        std::cout << "\nComparison total time: " << elapsed2.count() << " seconds" << std::endl;
        print_cmp_time_lsh();
        printf("==================================================\n");
    }

    time1 = std::chrono::high_resolution_clock::now();
    merge_union(rank, size); // Merge results across processes
    time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = time2 - time1;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Saving the final cleaned dataset.." << std::endl;
    }
    delete_hash_result(outputPath); // remove the binary hash result files
    time1 = std::chrono::high_resolution_clock::now();
    for (int i=start_index; i < end_index; i++) {
        const string &fp=file_list[i];
        generate_file(fp, i, outputPath);  // save the final deduplicated dataset 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed4 = time2 - time1;
    
    // Print total times for all phases
    if (rank == 0) {
        printf("==================================================\n");
        std::cout << "Min Hash total time: " << elapsed1.count() << " seconds" << std::endl;
        std::cout << "Comparison total time: " << elapsed2.count() << " seconds" << std::endl;
        std::cout << "Union total time: " << elapsed3.count() << " seconds" << std::endl;
        std::cout << "File generate time: " << elapsed4.count() << " seconds" << std::endl;
        std::cout << "Total time: " << elapsed1.count()+elapsed2.count()+elapsed3.count()+elapsed4.count() << " seconds" << std::endl;
        printf("==================================================");
    }
    finalize_lsh();
    MPI_Finalize();
    return 0;
}
