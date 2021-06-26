#pragma once
#include <unordered_map>
#include <string>

#include "cuda_runtime.h"

struct Parameters {
	
	double time_step;
	double time_end;
	double eps;
	double sigma;
	size_t threads_per_block;
	size_t part_out_freq;
	size_t vtk_out_freq;
	std::string part_input_file;
	std::string part_out_name;
	std::string vtk_out_name;
	double3 boundary_min;
	double3 boundary_max;
	double3 cells_dim;
	double r_cut;
	double r_skin;
	size_t bin_freq;
	size_t neighborlist_length;

	Parameters(std::unordered_map<std::string, std::string> params) {
		part_input_file = params["part_input_file"];
		time_step = std::stod(params["timestep_length"]);
		time_end = std::stod(params["time_end"]);
		eps = std::stod(params["epsilon"]);
		sigma = std::stod(params["sigma"]);
		part_out_freq = std::stoul(params["part_out_freq"]);
		part_out_name = params["part_out_name_base"];
		vtk_out_freq = std::stoul(params["vtk_out_freq"]);
		vtk_out_name = params["vtk_out_name_base"];
		threads_per_block = std::stoul(params["cl_workgroup_1dsize"]);
		boundary_min = make_double3(std::stod(params["x_min"]), std::stod(params["y_min"]), std::stod(params["z_min"]));
		boundary_max = make_double3(std::stod(params["x_max"]), std::stod(params["y_max"]), std::stod(params["z_max"]));
		cells_dim = make_double3(std::stod(params["x_n"]), std::stod(params["y_n"]), std::stod(params["z_n"]));
		r_cut = std::stod(params["r_cut"]);
		r_skin = std::stod(params["r_skin"]);
		bin_freq = std::stod(params["bin_freq"]);
		neighborlist_length = std::stod(params["neighborlist_length"]);
	}
};


struct Particle {

	Particle(double mass, double3 pos, double3 vel) :
		mass(mass), pos(pos), vel(vel) {}

	double mass;
	double3 pos;
	double3 vel;
};