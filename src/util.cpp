#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
// #include "nlohmann/json.hpp"
// using json = nlohmann::json;

using namespace std;
namespace fs = std::filesystem;

// Function to calculate the size of a file
int filesize1(const std::string& filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return (int)in.tellg(); 
}


// Retrieve a list of all files in the given folder
bool getFileList(vector<string> &file_list, const std::string& folderPath) {
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Cannot find folder: " << folderPath << std::endl;
        return false;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        file_list.push_back(entry.path());
    }
    
    return true;
    // for(auto &e : file_list) {
    //     iterateLines(e);
    // }
}

// Define maximum number of lines to handle
#define MAX_LINE 30000
