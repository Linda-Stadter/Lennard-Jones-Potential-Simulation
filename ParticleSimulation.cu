#include <map>
#include <string>
#include <cuda_runtime.h>
#include <fstream>

#include "file_manager.h"
#include "simulation_structs.h"
#include "helpers_math.h"
#include "helpers.h"

__device__ int3 calculate_3d_idx(double3 position, double3 cell_length, int3 cells_dim, double3 boundary_min) {
	int3 particle_idx = double3ToInt3((position - boundary_min) / cell_length);
		
	particle_idx.x = particle_idx.x % (int)cells_dim.x;
	particle_idx.y = particle_idx.y % (int)cells_dim.y;
	particle_idx.z = particle_idx.z % (int)cells_dim.z;

	return particle_idx;
}

__global__ void binning_cell_parallel(Particle* particles, int* cell_list, int* particle_list, size_t N, size_t num_particles, double3 boundary_min, double3 boundary_max, double3 cells_dim, double3 cell_length) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		// reset list
		cell_list[tid] = -1;

		// calculate cell coordinate from id
		// (determines index order in cell_list)
		int z_idx = tid / (cells_dim.x * cells_dim.y);
		int tid_z = tid - (z_idx * cells_dim.x * cells_dim.y);
		int y_idx = tid_z / cells_dim.x;
		int x_idx = tid_z % (int)cells_dim.x;
		int3 current_cell = make_int3(x_idx, y_idx, z_idx);


		for (int i = 0; i < num_particles; i++) {
			int3 cell = calculate_3d_idx(particles[i].pos, cell_length, double3ToInt3(cells_dim), boundary_min);

			if (cell == current_cell) {
				particle_list[i] = cell_list[tid];
				cell_list[tid] = i;
			}
		}
	}
}

__global__ void reset_binning_particle_parallel(int* cell_list, size_t N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		// reset list
		cell_list[tid] = -1;
	}
}


__device__ int3 boundary_conditions(int3 neighbor_cell_idx, int3 cells_dim) {
	// boundary conditions
	if (neighbor_cell_idx.x < 0) {
		neighbor_cell_idx.x += cells_dim.x;
	}
	if (neighbor_cell_idx.y < 0) {
		neighbor_cell_idx.y += cells_dim.y;
	}
	if (neighbor_cell_idx.z < 0) {
		neighbor_cell_idx.z += cells_dim.z;
	}
	if (neighbor_cell_idx.x >= cells_dim.x) { 
		neighbor_cell_idx.x -= cells_dim.x;
	}
	if (neighbor_cell_idx.y >= cells_dim.y) {
		neighbor_cell_idx.y -= cells_dim.y;
	}
	if (neighbor_cell_idx.z >= cells_dim.z) {
		neighbor_cell_idx.z -= cells_dim.z;
	}
	return neighbor_cell_idx;
}


__global__ void binning_particle_parallel(Particle* particles, int* cell_list, int* particle_list, size_t N, size_t num_particles, double3 boundary_min, double3 boundary_max, int3 cells_dim, double3 cell_length) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < num_particles) {
		Particle& particle = particles[tid];
		double3 domain = boundary_max - boundary_min;
		int3 particle_cell_idx = calculate_3d_idx(particle.pos, cell_length, cells_dim, boundary_min);

		// make flat index out of int3 index
		int particle_flat_cell_idx = particle_cell_idx.x + particle_cell_idx.y * cells_dim.x + particle_cell_idx.z * cells_dim.x * cells_dim.y;

		/*
		atomic:
		particle_list[tid] = cell_list[particle_flat_cell_idx];
		cell_list[particle_flat_cell_idx] = tid;
		*/

		int old = atomicExch(&cell_list[particle_flat_cell_idx], tid);
		particle_list[tid] = old;
	}
}

__global__ void pre_integration(Particle* particles, double3* force, double deltaTime, size_t N, double3 boundary_min, double3 boundary_max) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		double3 position = particle.pos + deltaTime * particle.vel + (force[tid] * deltaTime * deltaTime) / (2 * particle.mass);
		double3 domain = boundary_max - boundary_min;

		//check if particles left the domain
		position.x = (position.x < boundary_min.x) ? boundary_max.x - fmod(abs(position.x), abs(domain.x)) : position.x;
		position.x = (position.x > boundary_max.x) ? boundary_min.x + fmod(abs(position.x), abs(domain.x)) : position.x;

		position.y = (position.y < boundary_min.y) ? boundary_max.y - fmod(abs(position.y), abs(domain.y)) : position.y;
		position.y = (position.y > boundary_max.y) ? boundary_min.y + fmod(abs(position.y), abs(domain.y)) : position.y;

		position.z = (position.z < boundary_min.z) ? boundary_max.z - fmod(abs(position.z), abs(domain.z)) : position.z;
		position.z = (position.z > boundary_max.z) ? boundary_min.z + fmod(abs(position.z), abs(domain.z)) : position.z;

		particle.pos = position;
		particle.vel += force[tid] * deltaTime / (2 * particle.mass);
	}
}

__global__ void post_integration(Particle* particles, double3* force, double deltaTime, size_t N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		particle.vel += force[tid] * deltaTime / (2 * particle.mass);
	}
}

__device__ double3 calculate_force(Particle& particle, Particle& neighbor_particle, double3 domain, double eps, double s6, double r_cut) {
	double3 force_tmp{ 0.0, 0.0, 0.0 };
	double3 dist = particle.pos - neighbor_particle.pos;

	// correct dist as maximal distance is now 0.5*domain
	double3 correction = make_double3(domain.x * round(dist.x / domain.x), domain.y * round(dist.y / domain.y), domain.z * round(dist.z / domain.z));
	dist = dist - correction;
	double dist_norm_sq = norm_sq(dist); // use square norm for optimization
	if (dist_norm_sq <= r_cut * r_cut) {
		double t = 1.0 / dist_norm_sq;
		double f1 = 24 * eps * t; // multiply t instead of divide by dist_norm_sq
		double f2 = s6 * t * t * t;
		double f3 = 2 * f2 - 1;
		force_tmp += f1 * f2 * f3 * dist;
	}
	return force_tmp;
}

__global__ void calculate_forces_brute_force(Particle* particles, double3* force, double eps, double s6, size_t N, double3 boundary_min, 
											double3 boundary_max, double r_cut) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		double3 force_tmp{ 0.0, 0.0, 0.0 };
		double3 domain = boundary_max - boundary_min;

		for (int i = 0; i < N; i++) {
			if (i == tid)
				continue;
			// do force calculation
			force_tmp += calculate_force(particle, particles[i], domain, eps, s6, r_cut);
		}

		force[tid] = force_tmp;
	}
}


__global__ void calculate_forces_particle_binning(Particle* particles, double3* force, double eps, double s6, size_t N,
	double3 boundary_min, double3 boundary_max, double r_cut, double3 cell_length, int3 cells_dim, int* cell_list, int* particles_list) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		double3 force_tmp{ 0.0, 0.0, 0.0 };
		double3 domain = boundary_max - boundary_min;

		int3 particle_cell_idx = calculate_3d_idx(particle.pos, cell_length, cells_dim, boundary_min);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					int3 neighbor_cell_idx = particle_cell_idx + make_int3(x, y, z);

					// boundary conditions
					neighbor_cell_idx = boundary_conditions(neighbor_cell_idx, cells_dim);
					
					// make flat index out of int3 index
					int neighbor_flat_cell_idx = neighbor_cell_idx.x + neighbor_cell_idx.y * cells_dim.x + neighbor_cell_idx.z * cells_dim.x * cells_dim.y;

					int neighbor_particle_idx = cell_list[neighbor_flat_cell_idx];
					while (neighbor_particle_idx != -1) {
						//check if particle_idx is this particle
						if (neighbor_particle_idx == tid) {
							neighbor_particle_idx = particles_list[neighbor_particle_idx];
							continue;
						}
						Particle neighbor_particle = particles[neighbor_particle_idx];

						// do force calculation
						force_tmp += calculate_force(particle, neighbor_particle, domain, eps, s6, r_cut);

						// follow linked list
						neighbor_particle_idx = particles_list[neighbor_particle_idx];
					}
				}
			}
		}
		force[tid] = force_tmp;
	}
}

__global__ void calculate_forces_cell_binning(Particle* particles, double3* force, double eps, double s6, size_t N,
	double3 boundary_min, double3 boundary_max, double r_cut, double3 cell_length, int3 cells_dim, int* cell_list, int* particles_list, size_t num_particles) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		int first_cell_particle_idx = cell_list[tid];

		// no particles in this cell
		if (first_cell_particle_idx == -1) {
			return;
		}

		double3 domain = boundary_max - boundary_min;

		int z_idx = tid / (cells_dim.x * cells_dim.y);
		int tid_z = tid - (z_idx * cells_dim.x * cells_dim.y);
		int y_idx = tid_z / cells_dim.x;
		int x_idx = tid_z % (int)cells_dim.x;
		int3 current_cell = make_int3(x_idx, y_idx, z_idx);

		// reset force for each particle in the cell
		int cell_particle_idx = first_cell_particle_idx;
		while (cell_particle_idx != -1) {
			force[cell_particle_idx] = { 0,0,0 };
			cell_particle_idx = particles_list[cell_particle_idx];
		}

		// iterate over all neighbor cells
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					int3 neighbor_cell_idx = current_cell + make_int3(x, y, z);

					// check boundary conditions
					neighbor_cell_idx = boundary_conditions(neighbor_cell_idx, cells_dim);

					// make flat index out of int3 index
					int neighbor_flat_cell_idx = neighbor_cell_idx.x + neighbor_cell_idx.y * cells_dim.x + neighbor_cell_idx.z * cells_dim.x * cells_dim.y;
					int neighbor_particle_idx = cell_list[neighbor_flat_cell_idx];

					cell_particle_idx = first_cell_particle_idx;
					int first_neighbor_cell_particle_idx = neighbor_particle_idx;
					//iterate over all particles in cell_particle_idx
					while (cell_particle_idx != -1) {
						Particle& particle = particles[cell_particle_idx];
						// reset neighbor_particle_idx with first particle of the neighbor cell
						neighbor_particle_idx = first_neighbor_cell_particle_idx;

						// iterate over all neighbor particles
						while (neighbor_particle_idx != -1) {
							Particle neighbor_particle = particles[neighbor_particle_idx];

							//check if particle_idx is the current particle
							if (neighbor_particle_idx == cell_particle_idx) {
								neighbor_particle_idx = particles_list[neighbor_particle_idx];
								continue;
							}
							
							// do force calculation
							force[cell_particle_idx] += calculate_force(particle, neighbor_particle, domain, eps, s6, r_cut);

							// move on to next neighbor particle in this neighbor cell
							neighbor_particle_idx = particles_list[neighbor_particle_idx];
						}

						// follow linked list
						cell_particle_idx = particles_list[cell_particle_idx];
					}
				}
			}
		}
	}
}

__global__ void calculate_forces_neighbor_lists(Particle* particles, double3* force, double eps, double s6, size_t N, double3 domain, double r_cut,
												double3 cell_length, int3 cells_dim, int* cell_list, int* particles_list, int neighborlist_length, int* neighborlists, int* num_neighs) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		double3 force_tmp{ 0.0, 0.0, 0.0 };
	
		for (int i = 0; i < num_neighs[tid]; i++) {
			int neighbor_particle_idx = neighborlists[tid * neighborlist_length + i];
			Particle neighbor_particle = particles[neighbor_particle_idx];

			// do force calculation
			force_tmp += calculate_force(particle, neighbor_particle, domain, eps, s6, r_cut);
		}
		force[tid] = force_tmp;
	}
}

__global__ void neighbor_lists(Particle* particles, int* cell_list, int* particle_list, size_t N, double3 boundary_min, double3 boundary_max, int3 cells_dim, double3 cell_length, int neighborlist_length, int* neighborlists, int* num_neighs, int r_cut, int r_skin) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		int index = 0;
		Particle& particle = particles[tid];
		double3 domain = boundary_max - boundary_min;
		int3 particle_cell_idx = calculate_3d_idx(particle.pos, cell_length, cells_dim, boundary_min);
		
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					int3 neighbor_cell_idx = particle_cell_idx + make_int3(x, y, z);

					// boundary conditions
					neighbor_cell_idx = boundary_conditions(neighbor_cell_idx, cells_dim);
					
					// make flat index out of int3 index
					int neighbor_flat_cell_idx = neighbor_cell_idx.x + neighbor_cell_idx.y * cells_dim.x + neighbor_cell_idx.z * cells_dim.x * cells_dim.y;

					int neighbor_particle_idx = cell_list[neighbor_flat_cell_idx];
					while (neighbor_particle_idx != -1) {
						//check if particle_idx is this particle
						if (neighbor_particle_idx == tid) {
							neighbor_particle_idx = particle_list[neighbor_particle_idx];
							continue;
						}

						// skin:
						Particle neighbor_particle = particles[neighbor_particle_idx];
						double3 dist = particle.pos - neighbor_particle.pos;
						double3 correction = make_double3(domain.x * round(dist.x / domain.x), domain.y * round(dist.y / domain.y), domain.z * round(dist.z / domain.z)); 
						dist = dist - correction;
						double dist_norm = norm(dist);
						if (dist_norm <= r_cut + r_skin) {

							// add to neighbor list for current particle
							neighborlists[tid * neighborlist_length + index] = neighbor_particle_idx;
							index++;
							// check if neighbor list is full
							if (index == neighborlist_length)
								goto Lbreak_loops;
						}

						// follow linked list
						neighbor_particle_idx = particle_list[neighbor_particle_idx];
					}
				}
			}
		}
		Lbreak_loops:
		num_neighs[tid] = index;
	}
}

int main() {
	FileManager file_manager("./input", "./output", "blocks_big.par");

	Parameters params = file_manager.readParams();
	std::vector<Particle> particles = file_manager.readInitialState(params.part_input_file, params.boundary_min, params.boundary_max);
	size_t num_particles = particles.size();

	double3 domain = params.boundary_max - params.boundary_min;
	double s6 = pow(params.sigma, 6);

	// binning structures
	std::vector<int> particle_list(num_particles, -1);
	size_t num_cells = params.cells_dim.x * params.cells_dim.y * params.cells_dim.z;
	double3 cell_length = domain / params.cells_dim;

	std::vector<int> cell_list(num_cells, -1);

	// Allocate memory on device
	Particle* d_particles;
	double3 *d_force;
	int *d_particle_list, *d_cell_list;
	size_t bytes_vec = sizeof(double3) * num_particles;
	size_t bytes_struct = sizeof(Particle) * num_particles;
	size_t bytes_particle_list = sizeof(int) * num_particles;
	size_t bytes_cell_list = sizeof(int) * num_cells;
	checkError(cudaMalloc((void**)&d_particle_list, bytes_particle_list));
	checkError(cudaMalloc((void**)&d_cell_list, bytes_cell_list));
	checkError(cudaMalloc((void**)&d_particles, bytes_struct));
	checkError(cudaMalloc(&d_force, bytes_vec));

	// Copy data to GPU
	checkError(cudaMemcpy(d_particles, particles.data(), bytes_struct, cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_particle_list, particle_list.data(), bytes_particle_list, cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_cell_list, cell_list.data(), bytes_cell_list, cudaMemcpyHostToDevice));

	dim3 threads_per_block(params.threads_per_block);
	dim3 numBlocks((num_particles + threads_per_block.x - 1) / threads_per_block.x);
	dim3 numBlocksCell((num_cells + threads_per_block.x - 1) / threads_per_block.x);

	//binning_cell_parallel<<<numBlocksCell, threads_per_block>>>(d_particles, d_cell_list, d_particle_list, num_cells, num_particles, params.boundary_min, params.boundary_max, params.cells_dim, cell_length);
	// OR
	reset_binning_particle_parallel<<<numBlocksCell, threads_per_block>>>(d_cell_list, num_cells);
	binning_particle_parallel<<<numBlocks, threads_per_block>>>(d_particles, d_cell_list, d_particle_list, num_cells, num_particles, params.boundary_min, params.boundary_max, double3ToInt3(params.cells_dim), cell_length);
	checkError(cudaPeekAtLastError());
	checkError(cudaDeviceSynchronize());
	// DEBUG binning
	/*checkError(cudaMemcpy(particle_list.data(), d_particle_list, bytes_particle_list, cudaMemcpyDeviceToHost));
	checkError(cudaMemcpy(cell_list.data(), d_cell_list, bytes_cell_list, cudaMemcpyDeviceToHost));
	for (int i = 0; i < cell_list.size(); i++) {
		if (cell_list.at(i) == -1) continue;
        std::cout << "Cell " << i << ": " << cell_list.at(i) << std::endl;
    }
	for (int i = 0; i < particle_list.size(); i++) {
        std::cout << "Particle " << i << ": " << particle_list.at(i) << std::endl;
    }*/

	int *d_neighborlists, *d_num_neighs;
	checkError(cudaMalloc((void**)&d_neighborlists, sizeof(int) * num_particles * params.neighborlist_length));
	checkError(cudaMalloc((void**)&d_num_neighs, sizeof(int) * num_particles));
	neighbor_lists<<<numBlocks, threads_per_block>>>(d_particles, d_cell_list, d_particle_list, num_particles, params.boundary_min, params.boundary_max, 
													double3ToInt3(params.cells_dim), cell_length, params.neighborlist_length, d_neighborlists, d_num_neighs, params.r_cut, params.r_skin);
	checkError(cudaPeekAtLastError());
	checkError(cudaDeviceSynchronize());

	// Initialize forces
	calculate_forces_brute_force<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.eps, s6, num_particles, params.boundary_min, params.boundary_max, params.r_cut);
	checkError(cudaPeekAtLastError());
	checkError(cudaDeviceSynchronize());

	std::cout << "Simulation started ..." << std::endl;
	size_t iterations = (size_t) (params.time_end / params.time_step + 1);
	for (size_t i = 0; i < iterations; i++) {
		
		// Save simulation states at given frequncies
		if (i % params.part_out_freq == 0) {
			checkError(cudaMemcpy(particles.data(), d_particles, bytes_struct, cudaMemcpyDeviceToHost));
			file_manager.bufferOutput(particles);
		}
		if (i % params.vtk_out_freq == 0) {
			checkError(cudaMemcpy(particles.data(), d_particles, bytes_struct, cudaMemcpyDeviceToHost));
			file_manager.bufferVTK(particles);
		}

		// Verlet scheme: Integrate position, calculate forces, and integrate velocity
		pre_integration<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.time_step, num_particles, params.boundary_min, params.boundary_max);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());

		if (i % params.bin_freq == 0) {
			binning_cell_parallel<<<numBlocksCell, threads_per_block>>>(d_particles, d_cell_list, d_particle_list, num_cells, num_particles, params.boundary_min, params.boundary_max, params.cells_dim, cell_length);
			// OR
			//reset_binning_particle_parallel << <numBlocksCell, threads_per_block >> > (d_particles, d_cell_list, d_particle_list, num_cells, num_particles, params.boundary_min, params.boundary_max, params.cells_dim, cell_length);
			//binning_particle_parallel << <numBlocks, threads_per_block >> > (d_particles, d_cell_list, d_particle_list, num_cells, num_particles, params.boundary_min, params.boundary_max, double3ToInt3(params.cells_dim), cell_length);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());

			neighbor_lists << <numBlocks, threads_per_block >> > (d_particles, d_cell_list, d_particle_list, num_particles, params.boundary_min, params.boundary_max,
				double3ToInt3(params.cells_dim), cell_length, params.neighborlist_length, d_neighborlists, d_num_neighs, params.r_cut, params.r_skin);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());
		}

		// choose one:
		//calculate_forces_brute_force<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.eps, s6, num_particles, params.boundary_min, params.boundary_max, params.r_cut);
		//calculate_forces_particle_binning<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.eps, s6, num_particles, params.boundary_min, params.boundary_max, params.r_cut, cell_length, double3ToInt3(params.cells_dim), d_cell_list, d_particle_list);
		calculate_forces_cell_binning<<<numBlocksCell, threads_per_block>>>(d_particles, d_force, params.eps, s6, num_cells, params.boundary_min, params.boundary_max, params.r_cut, cell_length, double3ToInt3(params.cells_dim), d_cell_list, d_particle_list, num_particles);
		//calculate_forces_neighbor_lists<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.eps, s6, num_particles, domain, params.r_cut, cell_length, double3ToInt3(params.cells_dim), d_cell_list, d_particle_list, params.neighborlist_length, d_neighborlists, d_num_neighs);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());

		post_integration<<<numBlocks, threads_per_block>>>(d_particles, d_force, params.time_step, num_particles);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());

	}

	std::cout << "Simulation finished!" << std::endl;

	file_manager.write(params.part_out_name, params.vtk_out_name);

	checkError(cudaFree(d_force));
	checkError(cudaFree(d_particles));
}
