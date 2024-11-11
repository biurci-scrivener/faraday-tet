#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <Eigen/Dense>

// ChatGPT generated most of this. I have lost my shame

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> parsePLY(std::string filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error reading file" << filename << std::endl;
        exit(1);
    }

    std::string line;
    bool inHeader = true;
    int vertexCount = 0;

    // Read the header
    while (inHeader && std::getline(file, line)) {
        if (line == "end_header") {
            inHeader = false;
            continue;
        }

        if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string element;
            iss >> element >> element >> vertexCount;
        } 
    }

    Eigen::MatrixXd V(vertexCount, 3);
    Eigen::MatrixXd N(vertexCount, 3);

    // Read vertex data
    for (int i = 0; i < vertexCount; ++i) {
        
        double x, y, z;
        file >> x >> y >> z;
        V.row(i) << x, y, z;

        // Check for normals
        if (file.peek() != '\n') {
            double xn, yn, zn;
            file >> xn >> yn >> zn;
            N.row(i) << xn, yn, zn;
        }

    }

    file.close();

    return std::make_tuple(V, N);
}