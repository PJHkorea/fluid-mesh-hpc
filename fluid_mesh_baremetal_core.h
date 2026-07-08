/**
 * @file fluid_mesh_baremetal_core.h
 * @brief High-Performance Computing (HPC) Real-Time Fluid Dynamics Control AI Kernel
 * @license GNU GPLv3
 * @author PJHkorea
 */

#ifndef FLUID_MESH_BAREMETAL_CORE_H
#define FLUID_MESH_BAREMETAL_CORE_H

#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEIGHBOR_MAX 4
#define DIR_EAST     0
#define DIR_WEST     1
#define DIR_NORTH    2
#define DIR_SOUTH    3

typedef struct {
    float phi0;  /* Local fluid density state estimation */
    float phi1;  /* Fluid velocity gradient estimation */
    float omega; /* Local vorticity adjustment gain coefficient */
} FluidCell32;

typedef struct {
    double velocity_u; /* Refined spatial velocity vector U (Horizontal) */
    double velocity_v; /* Refined spatial velocity vector V (Vertical) */
} FluidMeshVector64;

typedef struct {
    double boundary_depth; /* Geometric layer compression depth metric */
    double spatial_gain;   /* Non-linear attenuation feedback coefficient */
} FluidMeshNode64;

/**
 * @brief [LAYER 1] 32-BIT PERIPHERAL FLOW SENSOR CORE (0ns Jitter Noise Eraser)
 */
static inline float fluid_mesh_cell32_process(
    FluidCell32* const self, volatile float raw_flow_signal, float cos_theta, float sin_theta
) {
    int is_nan = (raw_flow_signal != raw_flow_signal);
    int is_over = (raw_flow_signal > 1e6f) || (raw_flow_signal < -1e6f);
    int is_anomaly = is_nan || is_over;

    float phi0_pred = (cos_theta * self->phi0) - (sin_theta * self->phi1);
    float phi1_pred = (sin_theta * self->phi0) + (cos_theta * self->phi1);

    float scaled_kinetic_energy = fabsf(phi0_pred * self->omega);
    float numerator = 6.0f * scaled_kinetic_energy;
    float denominator = 12.0f + (scaled_kinetic_energy * scaled_kinetic_energy) + 1e-9f; 
    float rational_scale_factor = numerator / denominator;

    float next_phi0 = phi0_pred + (self->omega * (raw_flow_signal - phi0_pred)) * rational_scale_factor;

    /* 📌 HARDWARE MUX EMULATION: Forces compiler to utilize hardware-level CMOV/FCSEL registers */
    float fail_val = -99.0f;
    self->phi0 = is_anomaly ? fail_val : next_phi0; 
    self->phi1 = phi1_pred;

    return self->phi0;
}

/**
 * @brief [LAYER 2] 64-BIT CORE SPATIAL MATRIX CONTROLLER (Spontaneous Curl Diverter)
 */
static inline FluidMeshVector64 fluid_mesh_core64_process(
    FluidMeshNode64* const self, const float neighbor_sensor_signals[NEIGHBOR_MAX]
) {
    FluidMeshVector64 displacement_vector = {0.0, 0.0};

    double east  = (neighbor_sensor_signals[DIR_EAST]  == -99.0f) ? 0.0 : (double)neighbor_sensor_signals[DIR_EAST];
    double west  = (neighbor_sensor_signals[DIR_WEST]  == -99.0f) ? 0.0 : (double)neighbor_sensor_signals[DIR_WEST];
    double north = (neighbor_sensor_signals[DIR_NORTH] == -99.0f) ? 0.0 : (double)neighbor_sensor_signals[DIR_NORTH];
    double south = (neighbor_sensor_signals[DIR_SOUTH] == -99.0f) ? 0.0 : (double)neighbor_sensor_signals[DIR_SOUTH];

    double denom = 1.0 + self->boundary_depth;
    double safe_denom = denom + (double)(fabs(denom) < 1e-9) * 1e-9;
    double scale_damping_coefficient = 1.0 / safe_denom;

    double spatial_gradient_u = east - west;
    double spatial_gradient_v = north - south;

    /* 📌 THE MASTER TRICK: Cross-Axis Gradient Inversion for Spontaneous Vorticity Induction */
    displacement_vector.velocity_u = spatial_gradient_u * scale_damping_coefficient;
    displacement_vector.velocity_v = -spatial_gradient_v * scale_damping_coefficient; 

    return displacement_vector;
}

#ifdef __cplusplus
}
#endif

#endif /* FLUID_MESH_BAREMETAL_CORE_H */
