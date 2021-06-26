#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <experimental/filesystem>
#include <ios>
#include "helpers_math.h"

#include "simulation_structs.h"

namespace fs = std::experimental::filesystem;

class FileManager {

    std::string input_base;
    std::string output_base;
    std::string input_file;

    std::vector<std::vector<Particle>> output_buffer;
    std::vector<std::vector<Particle>> vtk_buffer;
    
public:

    FileManager(const std::string& input_base, const std::string& output_base, const std::string& input_file) : 
        input_base(input_base), output_base(output_base), input_file(input_file) {};

    Parameters readParams() {
        std::unordered_map<std::string, std::string> params;
        std::ifstream paramStream(input_base + "/" + input_file);
        std::string param, value;

        while (paramStream >> param >> value) {
            params[param] = value;
        }
        paramStream.close();

        return Parameters(params);
    }


    std::vector<Particle> readInitialState(const std::string& file_name, double3 boundary_min, double3 boundary_max) {
        std::vector<Particle> particles;
        int particleNumber;
        std::ifstream posStream(input_base + "/" + file_name);

        posStream >> particleNumber;
        double m, x, y, z, vx, vy, vz;
        double3 domain = boundary_max - boundary_min;

        while (posStream >> m >> x >> y >> z >> vx >> vy >> vz) {
            x = (x < boundary_min.x) ? boundary_max.x - fmod(abs(x), abs(domain.x)) : x;
            x = (x > boundary_max.x) ? boundary_min.x + fmod(abs(x), abs(domain.x)) : x;

            y = (y < boundary_min.y) ? boundary_max.y - fmod(abs(y), abs(domain.y)) : y;
            y = (y > boundary_max.y) ? boundary_min.y + fmod(abs(y), abs(domain.y)) : y;

            z = (z < boundary_min.z) ? boundary_max.z - fmod(abs(z), abs(domain.z)) : z;
            z = (z > boundary_max.z) ? boundary_min.z + fmod(abs(z), abs(domain.z)) : z;

            particles.emplace_back(m , make_double3(x, y, z), make_double3(vx, vy, vz));
        }
        posStream.close();

        return particles;
    }


    void bufferOutput(const std::vector<Particle>& particles) {
        output_buffer.push_back(particles);
    }

    void bufferVTK(const std::vector<Particle>& particles) {
        vtk_buffer.push_back(particles);
    }

    void write(std::string output_name, std::string vtk_name) {
        
        // Create output directory
        std::string subdir = input_file.substr(0, input_file.find("."));
        fs::create_directories(output_base + "/" + subdir);

        std::cout << "Writing files ..." << std::endl;
        
        // Write output files
        for (size_t i = 0; i < output_buffer.size(); i++) {
            std::ofstream o_stream(output_base + "/" + subdir + "/" + output_name + std::to_string(i) + ".out");
            o_stream << output_buffer[i].size() << std::endl;
            for (const auto& particle : output_buffer[i]) {
                o_stream << std::fixed << particle.mass << " " <<
                    particle.pos.x << " " << particle.pos.y << " " << particle.pos.z << " " <<
                    particle.vel.x << " " << particle.vel.y << " " << particle.vel.z << std::endl;
            }
            o_stream.close();
        }

        // Write VTK files
        for (size_t i = 0; i < vtk_buffer.size(); i++) {
            std::ofstream o_stream(output_base + "/" + subdir + "/" + vtk_name + std::to_string(i) + ".vtk");
            o_stream << "# vtk DataFile Version 4.0" << std::endl << "hesp visualization file" << std::endl << "ASCII" << std::endl << 
                "DATASET UNSTRUCTURED_GRID" << std::endl << "POINTS " << vtk_buffer[0].size() << " double" << std::endl;
            
            for (const auto& particle : vtk_buffer[i]) {
                o_stream << std::fixed << particle.pos.x << " " << particle.pos.y << " " << particle.pos.z << std::endl;
            }

            o_stream << "CELLS 0 0" << std::endl << "CELL_TYPES 0" << std::endl << "POINT_DATA " << vtk_buffer[0].size() << std::endl <<
                "SCALARS m double" << std::endl << "LOOKUP_TABLE default" << std::endl;

            for (const auto& particle : vtk_buffer[i]) {
                o_stream << std::fixed << particle.mass << std::endl;
            }

            o_stream << "VECTORS v double" << std::endl;

            for (const auto& particle : vtk_buffer[i]) {
                o_stream << std::fixed << particle.vel.x << " " << particle.vel.y << " " << particle.vel.z << std::endl;
            }

            o_stream.close();
        }

        std::cout << "Complete!" << std::endl;
    }
};

