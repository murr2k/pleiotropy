/*
 * CUDA Implementation of Factorization Algorithms
 * Optimized for NVIDIA GPUs to factorize 2539123152460219
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 1024
#define MAX_ITERATIONS 100000

typedef unsigned long long uint64_t;

/* Device function for modular multiplication with overflow protection */
__device__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = 0;
    a %= mod;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result + a) % mod;
        }
        a = (a * 2) % mod;
        b /= 2;
    }
    return result % mod;
}

/* Device function for GCD */
__device__ uint64_t gcd(uint64_t a, uint64_t b) {
    uint64_t temp;
    while (b != 0) {
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/* Device function for Pollard's rho polynomial f(x) = (x^2 + c) mod n */
__device__ uint64_t pollard_f(uint64_t x, uint64_t n, uint64_t c) {
    return (mulmod(x, x, n) + c) % n;
}

/* CUDA kernel for parallel Pollard's rho */
__global__ void pollard_rho_kernel(uint64_t n, uint64_t* factors, int* found) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread uses different random parameters
    curandState state;
    curand_init(tid, 0, 0, &state);
    
    uint64_t x = curand(&state) % (n - 2) + 2;
    uint64_t c = curand(&state) % (n - 1) + 1;
    uint64_t y = x;
    uint64_t d = 1;
    
    // Pollard's rho with product accumulation
    uint64_t product = 1;
    
    for (int i = 0; i < MAX_ITERATIONS && !(*found); i++) {
        x = pollard_f(x, n, c);
        y = pollard_f(pollard_f(y, n, c), n, c);
        
        uint64_t diff = (x > y) ? x - y : y - x;
        product = mulmod(product, diff, n);
        
        // Check GCD periodically
        if (i % 100 == 0) {
            d = gcd(product, n);
            if (d > 1 && d < n) {
                // Factor found! Try to atomically set it
                atomicCAS((unsigned long long*)found, 0, 1);
                factors[0] = d;
                factors[1] = n / d;
                return;
            }
            product = 1;
        }
    }
}

/* CUDA kernel for parallel trial division */
__global__ void trial_division_kernel(uint64_t n, uint64_t start, uint64_t end, 
                                     uint64_t* factor, int* found) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_threads = gridDim.x * blockDim.x;
    
    // Each thread checks a different range
    uint64_t chunk_size = (end - start) / total_threads;
    uint64_t my_start = start + tid * chunk_size;
    uint64_t my_end = (tid == total_threads - 1) ? end : my_start + chunk_size;
    
    // Make sure we start with an odd number
    if (my_start % 2 == 0) my_start++;
    
    for (uint64_t candidate = my_start; candidate < my_end && !(*found); candidate += 2) {
        if (n % candidate == 0) {
            atomicCAS((unsigned long long*)found, 0, 1);
            *factor = candidate;
            return;
        }
    }
}

/* CUDA kernel for ECM - Montgomery curve arithmetic */
__device__ void montgomery_add(uint64_t* x3, uint64_t* z3,
                               uint64_t x1, uint64_t z1,
                               uint64_t x2, uint64_t z2,
                               uint64_t xd, uint64_t zd,
                               uint64_t n) {
    uint64_t u = mulmod(x1 - z1 + n, x2 + z2, n);
    uint64_t v = mulmod(x1 + z1, x2 - z2 + n, n);
    uint64_t add = (u + v) % n;
    uint64_t sub = (u - v + n) % n;
    
    *x3 = mulmod(mulmod(add, add, n), zd, n);
    *z3 = mulmod(mulmod(sub, sub, n), xd, n);
}

__device__ void montgomery_double(uint64_t* x2, uint64_t* z2,
                                 uint64_t x, uint64_t z,
                                 uint64_t A, uint64_t n) {
    uint64_t s = (x + z) % n;
    uint64_t d = (x - z + n) % n;
    uint64_t s2 = mulmod(s, s, n);
    uint64_t d2 = mulmod(d, d, n);
    
    *x2 = mulmod(s2, d2, n);
    uint64_t t = (s2 - d2 + n) % n;
    *z2 = mulmod(t, (mulmod((A - 2) / 4, t, n) + d2) % n, n);
}

/* CUDA kernel for parallel ECM */
__global__ void ecm_kernel(uint64_t n, uint64_t B1, uint64_t* factors, int* found) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread uses a different curve
    curandState state;
    curand_init(tid, 0, 0, &state);
    
    uint64_t A = curand(&state) % (n - 1) + 1;
    uint64_t x = curand(&state) % (n - 1) + 1;
    uint64_t z = 1;
    
    // Stage 1: Compute k*P where k = lcm(1,2,...,B1)
    // Simplified version - in practice would use prime factorization
    for (uint64_t p = 2; p <= B1 && !(*found); p++) {
        // Skip if not prime (simplified check)
        if (p > 2 && p % 2 == 0) continue;
        
        uint64_t pp = p;
        while (pp <= B1) {
            // Scalar multiplication by p
            uint64_t x0 = x, z0 = z;
            for (int i = 1; i < p; i++) {
                montgomery_double(&x, &z, x, z, A, n);
                
                // Check for factor
                uint64_t d = gcd(z, n);
                if (d > 1 && d < n) {
                    atomicCAS((unsigned long long*)found, 0, 1);
                    factors[0] = d;
                    factors[1] = n / d;
                    return;
                }
            }
            pp *= p;
        }
    }
}

/* Host function to launch factorization */
extern "C" {
    void cuda_factorize(uint64_t n, uint64_t* factor1, uint64_t* factor2) {
        // Device memory
        uint64_t *d_factors;
        int *d_found;
        
        cudaMalloc(&d_factors, 2 * sizeof(uint64_t));
        cudaMalloc(&d_found, sizeof(int));
        cudaMemset(d_found, 0, sizeof(int));
        
        // Try different algorithms in sequence
        
        // 1. Trial division for small factors
        if (n < 1e12) {
            uint64_t limit = sqrt((double)n);
            trial_division_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, 3, limit, d_factors, d_found);
            cudaDeviceSynchronize();
            
            int found;
            cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found) {
                cudaMemcpy(factor1, d_factors, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                *factor2 = n / (*factor1);
                cudaFree(d_factors);
                cudaFree(d_found);
                return;
            }
        }
        
        // 2. Parallel Pollard's rho
        cudaMemset(d_found, 0, sizeof(int));
        pollard_rho_kernel<<<NUM_BLOCKS * 4, BLOCK_SIZE>>>(n, d_factors, d_found);
        cudaDeviceSynchronize();
        
        int found;
        cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (found) {
            cudaMemcpy(factor1, d_factors, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(factor2, d_factors + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaFree(d_factors);
            cudaFree(d_found);
            return;
        }
        
        // 3. ECM for harder cases
        cudaMemset(d_found, 0, sizeof(int));
        ecm_kernel<<<NUM_BLOCKS * 2, BLOCK_SIZE>>>(n, 50000, d_factors, d_found);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (found) {
            cudaMemcpy(factor1, d_factors, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(factor2, d_factors + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }
        
        cudaFree(d_factors);
        cudaFree(d_found);
    }
}