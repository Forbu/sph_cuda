// ============================================================================
//  asph_h100.cu — Anisotropic SPH atmospheric model, single H100
//
//  STATUS: NOT COMPILED, NOT RUN, NOT VALIDATED. Statically inspected only.
//  Run --quick and the audit gate before any long benchmark.
//
//  NOT A TOY — what is and isn't here:
//   INCLUDED: consistent summation density (lagged Shepard), hydrostatic
//     reference state mutually consistent to roundoff, buoyancy incl.
//     condensate loading, conserved-variable microphysics, bulk surface
//     flux, bulk viscosity with matching viscous heating, KDK leapfrog,
//     closed-loop CFL with abort trip, conservation audit.
//   DEFERRED TO NN (additive source terms only, never operators):
//     radiation SW/LW, ice/mixed-phase. Neither appears in SPH-side cost.
//   KNOWN LIMITATIONS, stated not hidden:
//     - difference-form pressure gradient is not exactly momentum-conserving
//     - viscosity is isotropic; horizontal mixing at 1 km under-resolved
//     - weakly-compressible perturbation form valid for Mach < ~0.15
//     - no radiation => runs limited to a few hours, no diurnal cycle
//     - theta updated by forward Euler while x,v use KDK (first-order scalar)
//
//  Build:
//    nvcc -O3 -use_fast_math -std=c++17 -arch=sm_90a --ptxas-options=-v \
//         -o asph_h100 asph_h100.cu -lcudart
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){                     \
    fprintf(stderr,"CUDA error %s:%d  %s\n",__FILE__,__LINE__,                 \
            cudaGetErrorString(_e)); exit(1);} }while(0)

// ---------------------------------------------------------------------------
//  Grid / kernel
// ---------------------------------------------------------------------------
constexpr float DX = 1000.0f, DY = 1000.0f, DZ = 100.0f;
constexpr float INV_DX = 1.0f/DX, INV_DY = 1.0f/DY, INV_DZ = 1.0f/DZ;
constexpr float V_CELL = DX*DY*DZ;

constexpr float H_SUP   = 1.0f;   // support = 2*H_SUP cells = 2 cells
constexpr int   STENCIL = 2;      // int(2*H_SUP); full spline sampled
constexpr float INV_H   = 1.0f/H_SUP;
constexpr float SIGMA   = 0.31830988618f;   // 1/pi, 3-D spline normalisation
constexpr float QMAX    = 2.0f;

// ---------------------------------------------------------------------------
//  Physical constants and reference sounding
//  Constant-N^2 profile: theta = th0*exp(N2 z/g), Pi = 1 - g^2/(cp th0 N2)
//  (1-exp(-N2 z/g)), p = P0 Pi^(R/cp), T = theta*Pi, rho = p/(R T).
//  These are mutually consistent and EXACTLY hydrostatic in the continuum:
//    dPi/dz = -g/(cp theta_r)  and  dp/dz = -rho g  (verified analytically).
// ---------------------------------------------------------------------------
constexpr float GRAV = 9.80665f;
constexpr float R_D  = 287.05f;
constexpr float CP   = 1004.0f;
constexpr float P0   = 100000.0f;
constexpr float L_V  = 2.501e6f;
constexpr float TH0  = 300.0f;      // surface potential temperature [K]
constexpr float N2   = 6.0e-5f;     // constant N^2 [1/s^2], N ~ 0.00775/s
constexpr float CS   = 80.0f;       // artificial sound speed [m/s]

constexpr float RHO0 = P0/(R_D*TH0);        // 1.1616 kg/m^3
constexpr float PMASS = RHO0*V_CELL;        // 1.1616e8 kg per particle (const)

// ---------------------------------------------------------------------------
//  Dissipation: gated Monaghan bulk viscosity. alpha is the physical
//  dissipation parameter; nu ~ alpha*c_s*h/8 ~ 400 m^2/s at alpha=0.2.
//  This IS the subgrid closure in this model. It is compressive-only, hence
//  horizontal shear mixing at 1 km is under-resolved -- a stated limitation.
// ---------------------------------------------------------------------------
constexpr float ALPHA_AV = 0.2f;
constexpr float H_AV     = 2.0f*DZ;         // 200 m dissipation length

// ---------------------------------------------------------------------------
//  Surface forcing: bulk aerodynamic formula, dry-ish land at noon.
//  |V|=7 m/s imposed large-scale wind, moisture availability beta=0.2,
//  skin-air dT ramping 0->12 K over 1 h. Gives H~121, LE~141 W/m^2,
//  Bowen ~0.86, BL encroachment depth ~1.09 km after 3 h, LCL ~1.1 km.
//  Self-limiting: as the BL warms, (T_s - T_air) falls and H drops.
// ---------------------------------------------------------------------------
constexpr float C_H     = 1.2e-3f;
constexpr float C_E     = 1.2e-3f;
constexpr float BETA_MOIST = 0.20f;
constexpr float V_BG    = 7.0f;
constexpr float DTS_SKIN = 12.0f;
constexpr float RAMP    = 3600.0f;

// ---------------------------------------------------------------------------
//  Moisture profile: near-constant RH ~70% in the lower troposphere, sharp
//  decay above 5 km. qv(0)=0.016 => RH 72%; consistent with the T profile.
// ---------------------------------------------------------------------------
constexpr float QV0 = 0.016f;

// ---------------------------------------------------------------------------
//  Warm-rain microphysics (ice deferred to NN)
// ---------------------------------------------------------------------------
constexpr float QC_AUTO  = 1.0e-3f;   // autoconversion threshold [kg/kg]
constexpr float RATE_AUTO= 1.0e-3f;   // [1/s]
constexpr float V_RAIN   = 9.0f;      // fall speed [m/s]
constexpr float THETA_CAP= 500.0f;
constexpr float QV_CAP   = 0.060f;

// ===========================================================================
//  Device helpers
// ===========================================================================
__device__ __forceinline__ float w3(float q){
    if (q >= QMAX) return 0.0f;
    if (q < 1.0f)  return SIGMA*(1.0f - 1.5f*q*q + 0.75f*q*q*q);
    float t = QMAX - q;  return 0.25f*SIGMA*t*t*t;
}
__device__ __forceinline__ float dw3(float q){          // dW/dq
    if (q >= QMAX) return 0.0f;
    if (q < 1.0f)  return SIGMA*(-3.0f*q + 2.25f*q*q);
    float t = QMAX - q;  return -0.75f*SIGMA*t*t;
}
// --- analytic reference state ----------------------------------------------
__device__ __forceinline__ float theta_ref(float z){
    return TH0*__expf(N2*z/GRAV);
}
__device__ __forceinline__ float exner_ref(float z){
    const float A = GRAV*GRAV/(CP*TH0*N2);
    return 1.0f - A*(1.0f - __expf(-N2*z/GRAV));
}
__device__ __forceinline__ float pres_ref(float z){
    return P0*__powf(exner_ref(z), CP/R_D);
}
__device__ __forceinline__ float rho_ref(float z){
    return pres_ref(z)/(R_D*theta_ref(z)*exner_ref(z));
}
__device__ __forceinline__ float qv_ref(float z){
    float q = QV0*__expf(-z/2500.0f);
    float d = fmaxf(z - 5000.0f, 0.0f);
    return q*__expf(-d/1000.0f);
}
// --- thermodynamics --------------------------------------------------------
__device__ __forceinline__ float esat(float T){
    float Tc = T - 273.15f;
    return 611.2f*__expf(17.67f*Tc/(Tc + 243.5f));
}
__device__ __forceinline__ float qsat(float T, float p){
    float es = fminf(esat(T), 0.5f*p);
    float den = p - 0.378f*es;
    return den > 1.0f ? 0.622f*es/den : 0.0f;
}
__device__ __forceinline__ float dqsdT(float T, float es, float p){
    // dq_s/dT for q_s = 0.622 e/(p - 0.378 e), e Magnus
    float Tc = T - 273.15f;
    float d  = Tc + 243.5f;
    float desdT = es*(17.67f*243.5f)/(d*d);
    float den = p - 0.378f*es;
    return 0.622f*p*desdT/(den*den);
}
__device__ __forceinline__ uint32_t part1by2_10(uint32_t n){
    n &= 0x000003ffu;
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n <<  8)) & 0x0300f00fu;
    n = (n ^ (n <<  4)) & 0x030c30c3u;
    n = (n ^ (n <<  2)) & 0x09249249u;
    return n;
}
__device__ __forceinline__ uint32_t morton3(uint32_t x,uint32_t y,uint32_t z){
    return part1by2_10(x) | (part1by2_10(y) << 1) | (part1by2_10(z) << 2);
}
__device__ __forceinline__ float hash_noise(int i,int j,int k){
    uint32_t h = (uint32_t)i*374761393u + (uint32_t)j*668265263u
               + (uint32_t)k*2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;  h ^= h >> 16;
    return (float)(h & 0x00ffffffu)*(1.0f/16777215.0f) - 0.5f;
}
__device__ __forceinline__ int cell_of(const float4& p,int nx,int ny,int nz){
    int ix = (int)floorf(p.x*INV_DX); ix = min(max(ix,0),nx-1);
    int iy = (int)floorf(p.y*INV_DY); iy = min(max(iy,0),ny-1);
    int iz = (int)floorf(p.z*INV_DZ); iz = min(max(iz,0),nz-1);
    return (iz*ny + iy)*nx + ix;
}
__device__ __forceinline__ void atomicMaxF(float* a, float v){
    atomicMax((unsigned int*)a, __float_as_uint(v));
}
// periodic wrap for x,y ; clamp for z
__device__ __forceinline__ int wrapx(int i,int nx){ return i<0 ? i+nx : (i>=nx ? i-nx : i); }
__device__ __forceinline__ int wrapy(int i,int ny){ return i<0 ? i+ny : (i>=ny ? i-ny : i); }

// ===========================================================================
//  INIT — lattice + analytic sounding. rho initialised to rho_ref so the
//  first Shepard normalisation returns rho_ref exactly.
// ===========================================================================
__global__ void kernel_init(float4* __restrict__ pos, float4* __restrict__ vel,
                            float4* __restrict__ thq, float* __restrict__ rho,
                            int nx,int ny,int nz,float noise)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = nx*ny*nz;
    if (i >= N) return;
    int ix = i % nx, iy = (i/nx) % ny, iz = i/(nx*ny);
    float x = (ix + 0.5f)*DX, y = (iy + 0.5f)*DY, z = (iz + 0.5f)*DZ;

    pos[i] = make_float4(x,y,z,0.f);
    vel[i] = make_float4(0.f,0.f,0.f,0.f);

    float th = theta_ref(z) + noise*hash_noise(ix,iy,iz);
    float qv = fmaxf(qv_ref(z)*(1.0f + 0.02f*hash_noise(ix+17,iy+31,iz+7)),0.f);
    thq[i] = make_float4(th, qv, 0.f, 0.f);
    rho[i] = rho_ref(z);
}

__global__ void kernel_morton_keys(const float4* __restrict__ pos,
                                   uint32_t* __restrict__ keys,
                                   int nx,int ny,int nz,int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 p = pos[i];
    uint32_t ix = (uint32_t)min(max((int)(p.x*INV_DX),0),nx-1);
    uint32_t iy = (uint32_t)min(max((int)(p.y*INV_DY),0),ny-1);
    uint32_t iz = (uint32_t)min(max((int)(p.z*INV_DZ),0),nz-1);
    keys[i] = morton3(ix,iy,iz);
}
__global__ void kernel_iota(uint32_t* __restrict__ v,int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) v[i] = (uint32_t)i;
}
__global__ void kernel_gather(const uint32_t* __restrict__ perm,
                              const float4* __restrict__ pA, float4* __restrict__ pB,
                              const float4* __restrict__ vA, float4* __restrict__ vB,
                              const float4* __restrict__ tA, float4* __restrict__ tB,
                              const float* __restrict__ rA, float* __restrict__ rB,
                              int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t s = __ldg(&perm[i]);
    pB[i] = pA[s]; vB[i] = vA[s]; tB[i] = tA[s]; rB[i] = rA[s];
}
// Morton-sorted => cells contiguous. Also rebuilds the surface-layer list.
__global__ void kernel_build_pairs(const float4* __restrict__ pos,
                                   int2* __restrict__ cellRange,
                                   int* __restrict__ surfList,
                                   int* __restrict__ surfCount,
                                   int nx,int ny,int nz,int N,int surfCap)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cid = cell_of(pos[i],nx,ny,nz);
    if (i == 0) cellRange[cid].x = 0;
    else {
        int pcid = cell_of(pos[i-1],nx,ny,nz);
        if (cid != pcid){ cellRange[cid].x = i; cellRange[pcid].y = i; }
    }
    if (i == N-1) cellRange[cid].y = N;

    if ((int)(pos[i].z*INV_DZ) == 0){
        int slot = atomicAdd(surfCount,1);
        if (slot < surfCap) surfList[slot] = i;
    }
}

// ===========================================================================
//  PASS 1 — summation density with lagged Shepard normalisation.
//  rho_i = SUM m W / SUM (m/rho_j^old) W  ; the ratio cancels the partition-
//  sum deficit at truncated support exactly when rho_j ~ rho_i. Interior
//  behaviour is a no-op; boundary deficit is removed. Lag error is O(dt).
// ===========================================================================
__global__ __launch_bounds__(256)
void kernel_density(const float4* __restrict__ pos,
                    const float*  __restrict__ rhoOld,
                    const int2*   __restrict__ cellRange,
                    float* __restrict__ rhoNew,
                    int nx,int ny,int nz,int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    int ix = min(max((int)(pi.x*INV_DX),0),nx-1);
    int iy = min(max((int)(pi.y*INV_DY),0),ny-1);
    int iz = min(max((int)(pi.z*INV_DZ),0),nz-1);

    bool interior = (iz>=STENCIL && iz<nz-STENCIL);
    int z0 = interior ? iz-STENCIL : max(iz-STENCIL,0);
    int z1 = interior ? iz+STENCIL : min(iz+STENCIL,nz-1);

    float S1 = 0.f, S2 = 0.f;
    for (int jz=z0; jz<=z1; ++jz){
        int basez = jz*ny;
        for (int jy=iy-STENCIL; jy<=iy+STENCIL; ++jy){
            int cy = wrapy(jy,ny);
            const int2* row = cellRange + (size_t)(basez+cy)*nx;
            for (int jx=ix-STENCIL; jx<=ix+STENCIL; ++jx){
                int2 rg = row[wrapx(jx,nx)];
                for (int j=rg.x; j<rg.y; ++j){
                    float4 pj = pos[j];
                    float dx = (pi.x-pj.x)*INV_DX;
                    float dy = (pi.y-pj.y)*INV_DY;
                    float dz = (pi.z-pj.z)*INV_DZ;
                    float r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 < 4.0f*H_SUP*H_SUP){
                        float W = w3(sqrtf(r2)*INV_H);
                        S1 += W;                       // mass factored out (const)
                        S2 += W/fmaxf(__ldg(&rhoOld[j]),1e-3f);
                    }
                }
            }
        }
    }
    // S1 = m*SUM W, S2 = m*SUM W/rho_j  =>  rho = S1/S2
    rhoNew[i] = S2 > 1e-20f ? S1/S2 : rho_ref(pi.z);
}

// ===========================================================================
//  PASS 2 — EOS, saturation adjustment, autoconversion.
//  Carries theta, qv, qc. Adjustment conserves theta_l = theta - L qc/(cp Pi)
//  and total water qv+qc exactly (Newton to tolerance, dqc clamped).
// ===========================================================================
__global__ void kernel_eos_micro(float4* __restrict__ thq,
                                 const float4* __restrict__ pos,
                                 const float* __restrict__ rho,
                                 float* __restrict__ prs,
                                 float* __restrict__ rainG,
                                 int nx,int ny,int nz,int N,float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p = pos[i];
    float4 t = thq[i];
    float z  = p.z;
    float r  = fmaxf(rho[i], 1e-3f);

    // perturbation pressure from the weakly-compressible EOS about the
    // analytic hydrostatic reference
    float ppr = CS*CS*(r - rho_ref(z));
    float pfl = fmaxf(pres_ref(z) + ppr, 1.0f);
    float Pi  = __powf(pfl/P0, R_D/CP);

    float th = t.x, qv = t.y, qc = t.z;
    float T  = th*Pi;

    float es = esat(T);
    float qs = qsat(T, pfl);

    if (qv > qs || qc > 0.f){
        float dqc = 0.f;                            // >0 condenses, <0 evaporates
        for (int it=0; it<3; ++it){
            float Tn  = T + (L_V/CP)*dqc;
            float esn = esat(Tn);
            float qsn = qsat(Tn, pfl);
            float f   = qv - dqc - qsn;
            float fp  = -1.f - dqsdT(Tn, esn, pfl)*(L_V/CP);
            dqc -= f/fp;
            dqc  = fminf(dqc, qv);                  // cannot condense more vapour
            dqc  = fmaxf(dqc, -qc);                 // cannot evaporate more cloud
        }
        qv -= dqc;  qc += dqc;
        th += L_V*dqc/(CP*Pi);                      // conserves theta_l
        T  += (L_V/CP)*dqc;
    }

    // autoconversion -> rain grid. Particle mass is constant (PMASS), so the
    // mass removed here equals the mass added to the grid: exact conservation.
    if (qc > QC_AUTO){
        float dqr = (qc - QC_AUTO)*(1.f - __expf(-RATE_AUTO*dt));
        qc -= dqr;
        int cid = cell_of(p,nx,ny,nz);
        atomicAdd(&rainG[cid], dqr*PMASS);          // kg into the cell
    }

    th = fminf(fmaxf(th,100.0f), THETA_CAP);
    qv = fminf(fmaxf(qv, 0.0f), QV_CAP);
    qc = fmaxf(qc, 0.0f);

    thq[i] = make_float4(th,qv,qc,0.f);
    prs[i] = ppr;
}

// ===========================================================================
//  PASS 3 — momentum.
//
//   a_i = -SUM_j m_j [ (p'_j - p'_i)/(rho_i rho_j) + PI_ij/(rho_i rho_j) ] gradW
//         + buoyancy  -  (difference form: first-order consistent, well
//                         conditioned; NOT exactly momentum-conserving)
//
//   Buoyancy includes condensate loading: g[(thv-thv_r)/thv_r - qc]
//   Viscous heating dtheta/dt = -(v . a_visc)/(cp Pi) is returned so the
//   kinetic energy removed by viscosity is returned as internal energy.
// ===========================================================================
__global__ __launch_bounds__(256)
void kernel_momentum(const float4* __restrict__ pos,
                     const float4* __restrict__ vel,
                     const float4* __restrict__ thq,
                     const float*  __restrict__ rho,
                     const float*  __restrict__ prs,
                     const int2*   __restrict__ cellRange,
                     float4* __restrict__ acc,
                     float*  __restrict__ dthdt,
                     int nx,int ny,int nz,int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i], vi = vel[i], ti = thq[i];
    int ix = min(max((int)(pi.x*INV_DX),0),nx-1);
    int iy = min(max((int)(pi.y*INV_DY),0),ny-1);
    int iz = min(max((int)(pi.z*INV_DZ),0),nz-1);

    bool interior = (iz>=STENCIL && iz<nz-STENCIL);
    int z0 = interior ? iz-STENCIL : max(iz-STENCIL,0);
    int z1 = interior ? iz+STENCIL : min(iz+STENCIL,nz-1);

    float rhoi = fmaxf(rho[i],1e-3f);
    float pri  = prs[i];

    float ax=0.f, ay=0.f, az=0.f;
    float avx=0.f, avy=0.f, avz=0.f;      // viscous part only (for heating)

    for (int jz=z0; jz<=z1; ++jz){
        int basez = jz*ny;
        for (int jy=iy-STENCIL; jy<=iy+STENCIL; ++jy){
            int cy = wrapy(jy,ny);
            const int2* row = cellRange + (size_t)(basez+cy)*nx;
            for (int jx=ix-STENCIL; jx<=ix+STENCIL; ++jx){
                int2 rg = row[wrapx(jx,nx)];
                for (int j=rg.x; j<rg.y; ++j){
                    if (j==i) continue;
                    float4 pj = pos[j];
                    float rx = pi.x-pj.x, ry = pi.y-pj.y, rz = pi.z-pj.z;
                    float dxt = rx*INV_DX, dyt = ry*INV_DY, dzt = rz*INV_DZ;
                    float r2t = dxt*dxt + dyt*dyt + dzt*dzt;
                    float lim = 4.0f*H_SUP*H_SUP;
                    if (r2t >= lim || r2t < 1e-12f) continue;

                    float q  = sqrtf(r2t)*INV_H;
                    // metric: dW_phys/dx_axis = [f'(q)/q]*dx_axis/(H^2*D_axis*V)
                    float k  = dw3(q)/q;                     // dW/dq / q
                    float gx = k*dxt*INV_DX;                 // = dW/dq*dxt/(q*DX) etc
                    float gy = k*dyt*INV_DY;
                    float gz = k*dzt*INV_DZ;

                    float rhoj = fmaxf(__ldg(&rho[j]),1e-3f);
                    float prj  = __ldg(&prs[j]);

                    // difference-form pressure term
                    float dp   = (prj - pri)/(rhoi*rhoj);

                    // Monaghan bulk viscosity, gated on approach
                    float4 vj  = vel[j];
                    float vdx  = vi.x-vj.x, vdy = vi.y-vj.y, vdz = vi.z-vj.z;
                    float vdotr= vdx*rx + vdy*ry + vdz*rz;
                    float visc = 0.f;
                    if (vdotr < 0.f){
                        float rp  = sqrtf(rx*rx+ry*ry+rz*rz);
                        float eta2= (0.01f*H_AV)*(0.01f*H_AV);
                        float mu  = vdotr/(rp*rp + eta2);      // [1/s]
                        float rbar= 0.5f*(rhoi+rhoj);
                        // bracket term must have units p/rho^2
                        visc = -ALPHA_AV*CS*H_AV*mu/(rbar*rbar*rhoj);
                    }
                    float br = dp + visc;

                    ax -= br*gx;  ay -= br*gy;  az -= br*gz;
                    avx-= visc*gx; avy-= visc*gy; avz-= visc*gz;
                }
            }
        }
    }
    // multiply by particle mass (constant) and the cell volume metric
    const float mm = PMASS/V_CELL;      // = RHO0
    ax*=mm; ay*=mm; az*=mm;
    avx*=mm; avy*=mm; avz*=mm;

    // buoyancy with condensate loading
    float thv    = ti.x*(1.0f + 0.608f*ti.y);
    float thvr   = theta_ref(pi.z)*(1.0f + 0.608f*qv_ref(pi.z));
    az += GRAV*((thv - thvr)/thvr - ti.z);

    acc[i]   = make_float4(ax,ay,az,0.f);
    // viscous heating: -v.a_visc/(cp Pi)  [K/s]
    float Pi = __powf(fmaxf(pres_ref(pi.z)+pri,1.f)/P0, R_D/CP);
    dthdt[i] = -(vi.x*avx + vi.y*avy + vi.z*avz)/(CP*Pi);
}

// ===========================================================================
//  KDK leapfrog — kick / drift. Called twice per step (half kicks).
// ===========================================================================
__global__ void kernel_kick(float4* __restrict__ vel,
                            const float4* __restrict__ acc,
                            int N,float hdt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 v = vel[i], a = acc[i];
    v.x += a.x*hdt; v.y += a.y*hdt; v.z += a.z*hdt;
    vel[i] = v;
}
__global__ void kernel_drift(float4* __restrict__ pos,
                             const float4* __restrict__ vel,
                             int N,float dt,float Lx,float Ly,float Lz)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 v = vel[i], p = pos[i];
    p.x += v.x*dt; p.y += v.y*dt; p.z += v.z*dt;
    // periodic x,y ; reflective z
    while (p.x < 0.f)  p.x += Lx;
    while (p.x >= Lx)  p.x -= Lx;
    while (p.y < 0.f)  p.y += Ly;
    while (p.y >= Ly)  p.y -= Ly;
    if (p.z < 0.f) { p.z = -p.z;      v.z = fabsf(v.z); vel[i]=v; }
    if (p.z > Lz)  { p.z = 2.f*Lz-p.z; v.z=-fabsf(v.z); vel[i]=v; }
    pos[i] = p;
}
__global__ void kernel_theta_update(float4* __restrict__ thq,
                                    const float* __restrict__ dthdt,
                                    int N,float dt)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 t = thq[i];
    t.x = fminf(fmaxf(t.x + dthdt[i]*dt, 100.0f), THETA_CAP);
    thq[i] = t;
}

// ===========================================================================
//  Rain: grid Eulerian upstream advection downward at V_RAIN.
//  Courant = V_RAIN*dt/DZ ~ 0.045 at dt=0.5s -> stable. Exact mass balance.
// ===========================================================================
__global__ void kernel_rain_adv(const float* __restrict__ rainIn,
                                float* __restrict__ rainOut,
                                float* __restrict__ precip2d,
                                int nx,int ny,int nz,float dt)
{
    int cid = blockIdx.x*blockDim.x + threadIdx.x;
    if (cid >= nx*ny*nz) return;
    int ix = cid % nx, iy = (cid/nx) % ny, iz = cid/(nx*ny);

    float cur = rainIn[cid];
    float frac = V_RAIN*dt*INV_DZ;
    if (frac > 1.f) frac = 1.f;
    float outMass = cur*frac;                 // leaves the bottom of this cell

    float inMass = 0.f;
    if (iz < nz-1) inMass = rainIn[cid + nx*ny]*frac;   // from the cell above

    rainOut[cid] = cur - outMass + inMass;

    if (iz == 0){
        atomicAdd(&precip2d[iy*nx+ix], outMass/(DX*DY));   // kg/m^2 accumulated
    }
}

// ===========================================================================
//  Noon surface flux — bulk aerodynamic formula, applied to particles
//  resident in the lowest layer (0..DZ). dtheta/dt = C_H |V| (Ts-Ta)/DZ
//  (density cancels); dq/dt = beta C_E |V| (qs(Ts,ps)-qv)/DZ. Self-limiting.
// ===========================================================================
__global__ void kernel_surface_flux(float4* __restrict__ thq,
                                    const float4* __restrict__ pos,
                                    const int* __restrict__ surfList,
                                    const int* __restrict__ surfCount,
                                    int surfCap,float dt,float Ts,float qs_sfc)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int n = min(*surfCount, surfCap);
    if (t >= n) return;
    int i = surfList[t];
    float4 p = pos[i];
    if (p.z >= DZ) return;                          // moved out of the layer
    float4 tq = thq[i];
    float Pi = __powf(fmaxf(pres_ref(p.z),1.f)/P0, R_D/CP);
    float Ta = tq.x*Pi;
    tq.x += C_H*V_BG*(Ts - Ta)*dt/DZ;
    tq.y += BETA_MOIST*C_E*V_BG*(qs_sfc - tq.y)*dt/DZ;
    tq.x  = fminf(fmaxf(tq.x,100.f),THETA_CAP);
    tq.y  = fminf(fmaxf(tq.y,0.f),QV_CAP);
    thq[i] = tq;
}

// ===========================================================================
//  Reductions: CFL velocity maxima, energy/water audit.
// ===========================================================================
__global__ void kernel_max_vel(const float4* __restrict__ vel,int N,
                               float* __restrict__ gmax)
{
    float mx=0.f,my=0.f,mz=0.f;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x){
        float4 v = vel[i];
        mx=fmaxf(mx,fabsf(v.x)); my=fmaxf(my,fabsf(v.y)); mz=fmaxf(mz,fabsf(v.z));
    }
    for (int o=16;o>0;o>>=1){
        mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));
        my=fmaxf(my,__shfl_down_sync(0xffffffffu,my,o));
        mz=fmaxf(mz,__shfl_down_sync(0xffffffffu,mz,o));
    }
    if ((threadIdx.x&31)==0){
        atomicMaxF(&gmax[0],mx); atomicMaxF(&gmax[1],my); atomicMaxF(&gmax[2],mz);
    }
}
// energy/water/momentum audit. Totals are per-particle sums scaled by PMASS.
// e[0]=KE, e[1]=PE, e[2]=internal (cp*theta*Pi), e[3]=water, e[4]=pz, e[5]=mass
__global__ void kernel_audit(const float4* __restrict__ pos,
                             const float4* __restrict__ vel,
                             const float4* __restrict__ thq,
                             double* __restrict__ tot,int N)
{
    double ke=0.0, pe=0.0, ie=0.0, w=0.0, pz=0.0;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x){
        float4 p=pos[i], v=vel[i], t=thq[i];
        float Pi = __powf(fmaxf(pres_ref(p.z),1.f)/P0, R_D/CP);
        ke += 0.5*(v.x*v.x+v.y*v.y+v.z*v.z);
        pe += GRAV*p.z;
        ie += CP*t.x*Pi;
        w  += (double)(t.y + t.z);
        pz += v.z;
    }
    double m = (double)PMASS;
    atomicAdd(&tot[0], ke*m); atomicAdd(&tot[1], pe*m); atomicAdd(&tot[2], ie*m);
    atomicAdd(&tot[3], w*m);  atomicAdd(&tot[4], pz*m); atomicAdd(&tot[5], m);
}
__global__ void kernel_clamp_vel(float4* __restrict__ vel,int N,float vmax){
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=gridDim.x*blockDim.x){
        float4 v=vel[i];
        float s2=v.x*v.x+v.y*v.y+v.z*v.z;
        if (s2>vmax*vmax){ float s=sqrtf(s2),f=vmax/s; v.x*=f;v.y*=f;v.z*=f; vel[i]=v; }
    }
}

// ===========================================================================
//  Diagnostics / output
// ===========================================================================
__global__ void kernel_coarse_dump(const int2* __restrict__ cellRange,
                                   const float4* __restrict__ vel,
                                   const float4* __restrict__ thq,
                                   float* __restrict__ out,
                                   int nx,int ny,int nz,
                                   int sx,int sy,int sz,int cx,int cy,int cz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= cx*cy*cz) return;
    int cxx=idx%cx, cyy=(idx/cx)%cy, czz=idx/(cx*cy);
    float a[6]={0,0,0,0,0,0}; int n=0;
    int ix1=min((cxx+1)*sx,nx), iy1=min((cyy+1)*sy,ny), iz1=min((czz+1)*sz,nz);
    for (int iz=czz*sz; iz<iz1; ++iz)
      for (int iy=cyy*sy; iy<iy1; ++iy)
        for (int ix=cxx*sx; ix<ix1; ++ix){
            int2 rg=cellRange[(size_t)(iz*ny+iy)*nx+ix];
            for (int j=rg.x;j<rg.y;++j){
                a[0]+=thq[j].x; a[1]+=thq[j].y; a[2]+=thq[j].z;
                a[3]+=vel[j].x; a[4]+=vel[j].y; a[5]+=vel[j].z; n++;
            }
        }
    float inv = n ? 1.f/(float)n : 0.f;
    for (int c=0;c<6;++c) out[(size_t)idx*6+c]=a[c]*inv;
}
__global__ void kernel_column_diag(const int2* __restrict__ cellRange,
                                   const float4* __restrict__ thq,
                                   const float4* __restrict__ vel,
                                   float4* __restrict__ coldiag,
                                   int nx,int ny,int nz)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col >= nx*ny) return;
    int iy=col/nx, ix=col%nx;
    float th=0.f,qv=0.f,qlmax=0.f,wmax=0.f; int n=0;
    for (int k=0;k<nz;++k){
        int2 rg=cellRange[(size_t)(k*ny+iy)*nx+ix];
        for (int j=rg.x;j<rg.y;++j){
            th+=thq[j].x; qv+=thq[j].y;
            qlmax=fmaxf(qlmax,thq[j].z); wmax=fmaxf(wmax,fabsf(vel[j].z)); n++;
        }
    }
    if (n){ float inv=1.f/(float)n; th*=inv; qv*=inv; }
    coldiag[col]=make_float4(th,qv,qlmax,wmax);
}