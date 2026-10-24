
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

// ===========================================================================
//  HOST
// ===========================================================================
struct Config {
    int nx=1000, ny=1000, nz=150;
    float dt_cap=0.f;
    float cfl=0.40f;
    int   cfl_interval=8;
    int   max_trip=20;
    bool  strict_cfl=false;
    double dump_every=900.0;
    double sim_hours=3.0;
    int dump_sx=10, dump_sy=10, dump_sz=3;
    std::string outdir="dumps";
    int report=200;
    bool validate=false;
    float noise=0.1f;
};
struct DumpHeader {
    uint32_t magic, version;
    int32_t nx,ny,nz,cx,cy,cz,sx,sy,sz,nfields;
    float t_sim,dt,lx,ly,lz;
};
static double now_s(){
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc,char**argv)
{
    Config cfg;
    for (int i=1;i<argc;++i){
        std::string a=argv[i];
        auto next=[&]{ return std::string(argv[++i]); };
        if      (a=="--nx") cfg.nx=atoi(next().c_str());
        else if (a=="--ny") cfg.ny=atoi(next().c_str());
        else if (a=="--nz") cfg.nz=atoi(next().c_str());
        else if (a=="--dt") cfg.dt_cap=atof(next().c_str());
        else if (a=="--cfl") cfg.cfl=atof(next().c_str());
        else if (a=="--cfl-interval") cfg.cfl_interval=atoi(next().c_str());
        else if (a=="--max-trip") cfg.max_trip=atoi(next().c_str());
        else if (a=="--strict-cfl") cfg.strict_cfl=true;
        else if (a=="--dump-every") cfg.dump_every=atof(next().c_str());
        else if (a=="--sim-hours") cfg.sim_hours=atof(next().c_str());
        else if (a=="--dump-stride"){ cfg.dump_sx=atoi(next().c_str());
                                      cfg.dump_sy=cfg.dump_sx;
                                      cfg.dump_sz=atoi(next().c_str()); }
        else if (a=="--outdir") cfg.outdir=next();
        else if (a=="--noise") cfg.noise=atof(next().c_str());
        else if (a=="--validate") cfg.validate=true;
        else if (a=="--quick"){ cfg.nx=128; cfg.ny=128; cfg.nz=64;
                                cfg.sim_hours=0.25; cfg.dump_every=300.0; }
        else if (a=="--help"){
            printf("asph_h100 [--nx N --ny N --nz N --dt cap --cfl c\n"
                   "          --cfl-interval k --max-trip n --strict-cfl\n"
                   "          --dump-every s --sim-hours h --dump-stride sx sz\n"
                   "          --outdir dir --noise a --validate --quick]\n");
            return 0;
        }
    }
    if (cfg.nx>1024||cfg.ny>1024||cfg.nz>1024){
        fprintf(stderr,"ERROR: 30-bit Morton key addresses 1024 cells/axis.\n");
        return 1;
    }

    const int N = cfg.nx*cfg.ny*cfg.nz;
    const size_t n4 = (size_t)N;
    const float Lx = cfg.nx*DX, Ly = cfg.ny*DY, Lz = cfg.nz*DZ;

    const float dt_static = cfg.cfl*fminf(fminf(DX,DY),DZ)/CS;
    const float dt_cap = cfg.dt_cap>0.f ? fminf(cfg.dt_cap,dt_static) : dt_static;
    const float dt_floor = 0.05f*dt_cap;
    float dt = dt_cap;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("=== ASPH-H100 =====================================================\n");
    printf("device        : %s (%d SM, %.1f GB)\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem/1e9);
    printf("STATUS        : NOT COMPILED-TESTED BY THE AUTHOR. Run --quick and\n");
    printf("                the audit gate before any long benchmark.\n");
    printf("domain        : %.0f x %.0f x %.0f km  periodic xy, reflective z\n",
           Lx/1e3,Ly/1e3,Lz/1e3);
    printf("parcels       : %d x %d x %d = %d  (mass %.3e kg each)\n",
           cfg.nx,cfg.ny,cfg.nz,N,PMASS);
    printf("sounding      : theta0=%.0f K, N^2=%.1e /s^2, c_s=%.0f m/s\n",
           TH0,N2,CS);
    printf("--- CFL -----------------------------------------------------------\n");
    printf("  static bound : dt <= %.4f s  (CFL %.2f * min(dx,dy,dz)/c_s)\n",
           dt_static,cfg.cfl);
    printf("  closed loop  : dt <= CFL*Daxis/(c_s+|v|max)*0.9 every %d steps\n",
           cfg.cfl_interval);
    printf("  floor %.4f s ; abort after %d violations%s\n", dt_floor,
           cfg.max_trip, cfg.strict_cfl?" (STRICT)":"");
    printf("--- FORCING -------------------------------------------------------\n");
    printf("  bulk flux    : C_H=C_E=%.1e |V|=%.0f m/s beta=%.2f dTs=%.0f K\n",
           C_H,V_BG,BETA_MOIST,DTS_SKIN);
    printf("  dtheta/dt    : %.2e K/s peak (= %.2f K/h in lowest %.0f m)\n",
           C_H*V_BG*DTS_SKIN/DZ, C_H*V_BG*DTS_SKIN/DZ*3600.0f, DZ);
    printf("  run          : %.2f sim h, dump every %.0f sim s\n",
           cfg.sim_hours,cfg.dump_every);
    printf("===================================================================\n");

    // ---- allocation -------------------------------------------------------
    float4 *posA,*velA,*thqA,*posB,*velB,*thqB;
    float *rhoCur,*rhoAlt,*prs,*dthdt,*rainA,*rainB,*precip2d,*coarse,*gmax;
    float4 *acc,*coldiag;
    uint32_t *keysIn,*keysOut,*valsIn,*perm;
    int2 *cellRange; int *surfList,*surfCount;
    void *sortTmp=nullptr; size_t sortTmpBytes=0;

    CK(cudaMalloc(&posA,n4*sizeof(float4))); CK(cudaMalloc(&velA,n4*sizeof(float4)));
    CK(cudaMalloc(&thqA,n4*sizeof(float4)));
    CK(cudaMalloc(&posB,n4*sizeof(float4))); CK(cudaMalloc(&velB,n4*sizeof(float4)));
    CK(cudaMalloc(&thqB,n4*sizeof(float4)));
    CK(cudaMalloc(&rhoCur,n4*sizeof(float))); CK(cudaMalloc(&rhoAlt,n4*sizeof(float)));
    CK(cudaMalloc(&prs,n4*sizeof(float)));   CK(cudaMalloc(&dthdt,n4*sizeof(float)));
    CK(cudaMalloc(&rainA,n4*sizeof(float)));  CK(cudaMalloc(&rainB,n4*sizeof(float)));
    CK(cudaMalloc(&acc,n4*sizeof(float4)));
    CK(cudaMalloc(&keysIn,n4*sizeof(uint32_t))); CK(cudaMalloc(&keysOut,n4*sizeof(uint32_t)));
    CK(cudaMalloc(&valsIn,n4*sizeof(uint32_t))); CK(cudaMalloc(&perm,n4*sizeof(uint32_t)));
    CK(cudaMalloc(&cellRange,n4*sizeof(int2)));
    const int surfCap = cfg.nx*cfg.ny*2;
    CK(cudaMalloc(&surfList,(size_t)surfCap*sizeof(int)));
    CK(cudaMalloc(&surfCount,sizeof(int)));
    CK(cudaMalloc(&precip2d,(size_t)cfg.nx*cfg.ny*sizeof(float)));
    CK(cudaMalloc(&coldiag,(size_t)cfg.nx*cfg.ny*sizeof(float4)));
    CK(cudaMalloc(&gmax,3*sizeof(float)));

    int cx=(cfg.nx+cfg.dump_sx-1)/cfg.dump_sx;
    int cy=(cfg.ny+cfg.dump_sy-1)/cfg.dump_sy;
    int cz=(cfg.nz+cfg.dump_sz-1)/cfg.dump_sz;
    size_t coarseBytes=(size_t)cx*cy*cz*6*sizeof(float);
    CK(cudaMalloc(&coarse,coarseBytes));
    float *h_coarse=nullptr; CK(cudaMallocHost(&h_coarse,coarseBytes));
    float4 *h_coldiag=nullptr;
    CK(cudaMallocHost(&h_coldiag,(size_t)cfg.nx*cfg.ny*sizeof(float4)));
    float *h_precip=nullptr;
    CK(cudaMallocHost(&h_precip,(size_t)cfg.nx*cfg.ny*sizeof(float)));
    float h_gmax[3];
    double h_tot[6], h_tot0[6];

    auto ceil_log2=[](int n){ int b=1; while((1<<b)<n) ++b; return b; };
    int bits = std::max(ceil_log2(cfg.nx),std::max(ceil_log2(cfg.ny),ceil_log2(cfg.nz)));
    int endbit = 3*bits;
    CK(cub::DeviceRadixSort::SortPairs(nullptr,sortTmpBytes,
        keysIn,keysOut,valsIn,perm,N,0,endbit));
    CK(cudaMalloc(&sortTmp,sortTmpBytes));
    printf("radix sort    : %d bits, temp %.2f GB\n", endbit, sortTmpBytes/1e9);
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
    printf("VRAM          : %.2f GB used of %.2f GB\n",(totB-freeB)/1e9,totB/1e9);
    mkdir(cfg.outdir.c_str(),0775);

    cudaFuncSetAttribute((const void*)kernel_density,
        cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute((const void*)kernel_momentum,
        cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxL1);

    const int TPB=256;
    const int BPB=(N+TPB-1)/TPB;
    CK(cudaMemset(cellRange,0,n4*sizeof(int2)));
    CK(cudaMemset(acc,0,n4*sizeof(float4)));
    CK(cudaMemset(rainA,0,n4*sizeof(float)));
    CK(cudaMemset(rainB,0,n4*sizeof(float)));
    CK(cudaMemset(precip2d,0,(size_t)cfg.nx*cfg.ny*sizeof(float)));
    CK(cudaMemset(dthdt,0,n4*sizeof(float)));

    kernel_init<<<BPB,TPB>>>(posA,velA,thqA,rhoCur,cfg.nx,cfg.ny,cfg.nz,cfg.noise);
    CK(cudaDeviceSynchronize());
    kernel_iota<<<BPB,TPB>>>(valsIn,N);

    float4 *pos=posA,*vel=velA,*thq=thqA,*pos2=posB,*vel2=velB,*thq2=thqB;
    float *rhoIn=rhoCur,*rhoOut=rhoAlt;
    float *rainIn=rainA,*rainOut=rainB;

    // ---- initial sort -----------------------------------------------------
    CK(cudaMemsetAsync(cellRange,0,n4*sizeof(int2)));
    CK(cudaMemsetAsync(surfCount,0,sizeof(int)));
    kernel_morton_keys<<<BPB,TPB>>>(pos,keysIn,cfg.nx,cfg.ny,cfg.nz,N);
    CK(cub::DeviceRadixSort::SortPairs(sortTmp,sortTmpBytes,keysIn,keysOut,
                                       valsIn,perm,N,0,endbit));
    kernel_gather<<<BPB,TPB>>>(perm,pos,pos2,vel,vel2,thq,thq2,rhoIn,rhoOut,N);
    std::swap(pos,pos2); std::swap(vel,vel2); std::swap(thq,thq2);
    std::swap(rhoIn,rhoOut);          // rhoIn now holds the gathered density
    CK(cudaMemsetAsync(cellRange,0,n4*sizeof(int2)));
    CK(cudaMemsetAsync(surfCount,0,sizeof(int)));
    kernel_build_pairs<<<BPB,TPB>>>(pos,cellRange,surfList,surfCount,
                                    cfg.nx,cfg.ny,cfg.nz,N,surfCap);
    // density: read rhoIn, write rhoOut, then swap so rhoIn = current
    kernel_density<<<BPB,TPB>>>(pos,rhoIn,cellRange,rhoOut,cfg.nx,cfg.ny,cfg.nz,N);
    std::swap(rhoIn,rhoOut);
    CK(cudaStreamSynchronize(0));

    // ---- initial force evaluation (needed for KDK) ------------------------
    kernel_eos_micro<<<BPB,TPB>>>(thq,pos,rhoIn,prs,rainIn,cfg.nx,cfg.ny,cfg.nz,dt);
    kernel_momentum<<<BPB,TPB>>>(pos,vel,thq,rhoIn,prs,cellRange,acc,dthdt,
                                 cfg.nx,cfg.ny,cfg.nz,N);
    CK(cudaStreamSynchronize(0));

    // initial audit
    CK(cudaMemset(h_tot,0,sizeof(h_tot)));
    { double* d_tot=nullptr; CK(cudaMalloc(&d_tot,6*sizeof(double)));
      CK(cudaMemset(d_tot,0,6*sizeof(double)));
      kernel_audit<<<1024,TPB>>>(pos,vel,thq,d_tot,N);
      CK(cudaMemcpy(h_tot0,d_tot,6*sizeof(double),cudaMemcpyDeviceToHost));
      CK(cudaFree(d_tot)); }

    // ---- timers -----------------------------------------------------------
    enum { T_SORT,T_DENS,T_EOS,T_MOM,T_INT,T_FLUX,T_RAIN,T_DUMP,T_CFL,T_COUNT };
    const char* tnames[T_COUNT]={"sort","density","eos+micro","momentum",
                                 "integrate","flux","rain","dump","cfl"};
    double tacc[T_COUNT]={0};

    double simt=0.0, sim_end=cfg.sim_hours*3600.0;
    double next_dump=cfg.dump_every;
    double since_sort=1e9;
    float  sort_dt=0.5f;
    float  dt_required=dt;
    int    step=0,trips=0,dumps=0,sort_count=0;
    double wall0=now_s(),wall_report=wall0,sim_report=0.0;
    bool   prof=false;

    auto tic=[&](){ if(prof) CK(cudaEventRecord(evA)); };
    cudaEvent_t evA,evB;
    CK(cudaEventCreate(&evA)); CK(cudaEventCreate(&evB));
    auto toc=[&](int slot){
        if(!prof) return;
        CK(cudaEventRecord(evB)); CK(cudaEventSynchronize(evB));
        float ms=0.f; CK(cudaEventElapsedTime(&ms,evA,evB));
        tacc[slot]+=ms/1000.0;
    };

    printf("\n%-8s %-9s %-8s %-9s %s\n","step","sim time","wall s","dt s","CFL / cloud / precip");

    while (simt < sim_end){
        ++step;
        prof = (step % 50 == 0);

        // ---- 1. sort (displacement-gated) --------------------------------
        if (since_sort >= sort_dt){
            tic();
            kernel_morton_keys<<<BPB,TPB>>>(pos,keysIn,cfg.nx,cfg.ny,cfg.nz,N);
            CK(cub::DeviceRadixSort::SortPairs(sortTmp,sortTmpBytes,keysIn,keysOut,
                                               valsIn,perm,N,0,endbit));
            kernel_gather<<<BPB,TPB>>>(perm,pos,pos2,vel,vel2,thq,thq2,rhoIn,rhoOut,N);
            std::swap(pos,pos2); std::swap(vel,vel2); std::swap(thq,thq2);
            std::swap(rhoIn,rhoOut);
            CK(cudaMemsetAsync(cellRange,0,n4*sizeof(int2)));
            CK(cudaMemsetAsync(surfCount,0,sizeof(int)));
            kernel_build_pairs<<<BPB,TPB>>>(pos,cellRange,surfList,surfCount,
                                            cfg.nx,cfg.ny,cfg.nz,N,surfCap);
            toc(T_SORT);
            since_sort=0.0; ++sort_count;
        }

        // ---- 2. first half kick ------------------------------------------
        tic();
        kernel_kick<<<BPB,TPB>>>(vel,acc,N,0.5f*dt);

        // ---- 3. drift ----------------------------------------------------
        kernel_drift<<<BPB,TPB>>>(pos,vel,N,dt,Lx,Ly,Lz);
        toc(T_INT);

        // ---- 4. surface flux ---------------------------------------------
        {
            float r = fminf(fmaxf((float)simt/RAMP,0.f),1.f);
            float sw = r*r*(3.f-2.f*r);
            float Ts = TH0 + DTS_SKIN*sw;              // prescribed skin temperature
            float qs_sfc = qsat(Ts, pres_ref(0.f));
            tic();
            int cb=(surfCap+TPB-1)/TPB;
            kernel_surface_flux<<<cb,TPB>>>(thq,pos,surfList,surfCount,surfCap,dt,
                                            Ts,qs_sfc);
            toc(T_FLUX);
        }

        // ---- 5. density (Shepard) ----------------------------------------
        tic();
        kernel_density<<<BPB,TPB>>>(pos,rhoIn,cellRange,rhoOut,cfg.nx,cfg.ny,cfg.nz,N);
        std::swap(rhoIn,rhoOut);
        toc(T_DENS);

        // ---- 6. EOS + microphysics ---------------------------------------
        tic();
        kernel_eos_micro<<<BPB,TPB>>>(thq,pos,rhoIn,prs,rainIn,cfg.nx,cfg.ny,cfg.nz,dt);
        toc(T_EOS);

        // ---- 7. momentum -------------------------------------------------
        tic();
        kernel_momentum<<<BPB,TPB>>>(pos,vel,thq,rhoIn,prs,cellRange,acc,dthdt,
                                     cfg.nx,cfg.ny,cfg.nz,N);
        toc(T_MOM);

        // ---- 8. theta update (viscous heating) ---------------------------
        tic();
        kernel_theta_update<<<BPB,TPB>>>(thq,dthdt,N,dt);
        toc(T_INT);

        // ---- 9. second half kick -----------------------------------------
        kernel_kick<<<BPB,TPB>>>(vel,acc,N,0.5f*dt);

        // ---- 10. rain advection ------------------------------------------
        tic();
        {
            int cb=(cfg.nx*cfg.ny*cfg.nz+TPB-1)/TPB;
            kernel_rain_adv<<<cb,TPB>>>(rainIn,rainOut,precip2d,
                                        cfg.nx,cfg.ny,cfg.nz,dt);
            std::swap(rainIn,rainOut);
        }
        toc(T_RAIN);

        simt += dt;  since_sort += dt;

        // ---- 11. CLOSED-LOOP CFL ----------------------------------------
        if (step % cfg.cfl_interval == 0){
            tic();
            CK(cudaMemsetAsync(gmax,0,3*sizeof(float)));
            kernel_max_vel<<<1024,TPB>>>(vel,N,gmax);
            CK(cudaMemcpyAsync(h_gmax,gmax,3*sizeof(float),cudaMemcpyDeviceToHost));
            CK(cudaStreamSynchronize(0));
            toc(T_CFL);

            float ux=h_gmax[0],uy=h_gmax[1],uz=h_gmax[2];
            dt_required = cfg.cfl*fminf(fminf(DX/(CS+ux),DY/(CS+uy)),
                                        DZ/(CS+uz))*0.90f;
            float adopt = fminf(dt_cap,dt_required);

            if (adopt < dt_floor){
                ++trips;
                fprintf(stderr,
                  "[CFL] step %d t=%.1fs : required %.4f s < floor %.4f s "
                  "(|u|=%.1f |v|=%.1f |w|=%.1f; binding %s)\n",
                  step,simt,adopt,dt_floor,ux,uy,uz,
                  (DZ/(CS+uz)<=fminf(DX/(CS+ux),DY/(CS+uy)))?"z":"xy");
                if (cfg.strict_cfl || trips > cfg.max_trip){
                    fprintf(stderr,
                      "[CFL] ABORT after %d violation(s). The scheme is outside "
                      "its stability envelope.\n",trips);
                    return 2;
                }
                kernel_clamp_vel<<<1024,TPB>>>(vel,N,0.5f*CS);
                adopt = dt_floor;
            } else if (adopt < 0.98f*dt){ trips=0; }
            dt = adopt;

            float dz_move = 0.3f*DZ/fmaxf(uz,0.5f);
            float dh_move = 0.3f*DX/fmaxf(fmaxf(ux,uy),0.5f);
            sort_dt = fminf(fmaxf(fminf(dz_move,dh_move),(float)dt),15.0f);
        }

        // ---- 12. dump ----------------------------------------------------
        if (simt >= next_dump){
            int ct=(cx*cy*cz+TPB-1)/TPB;
            int cb=(cfg.nx*cfg.ny+TPB-1)/TPB;
            tic();
            kernel_coarse_dump<<<ct,TPB>>>(cellRange,vel,thq,coarse,cfg.nx,cfg.ny,cfg.nz,
                                           cfg.dump_sx,cfg.dump_sy,cfg.dump_sz,cx,cy,cz);
            kernel_column_diag<<<cb,TPB>>>(cellRange,thq,vel,coldiag,cfg.nx,cfg.ny,cfg.nz);
            CK(cudaMemcpy(h_coarse,coarse,coarseBytes,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_coldiag,coldiag,(size_t)cfg.nx*cfg.ny*sizeof(float4),
                          cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_precip,precip2d,(size_t)cfg.nx*cfg.ny*sizeof(float),
                          cudaMemcpyDeviceToHost));
            CK(cudaStreamSynchronize(0));

            int ncol=cfg.nx*cfg.ny,ncloud=0;
            float qlmax=0.f,wmax=0.f,psum=0.f;
            for (int c=0;c<ncol;++c){
                float4 d=h_coldiag[c];
                if (d.z>1e-4f) ++ncloud;
                qlmax=fmaxf(qlmax,d.z); wmax=fmaxf(wmax,d.w); psum+=h_precip[c];
            }
            float cloudcov = ncol?100.f*(float)ncloud/ncol:0.f;
            double mm = psum/ncol;
            CK(cudaMemsetAsync(precip2d,0,(size_t)cfg.nx*cfg.ny*sizeof(float)));

            char fn[512];
            snprintf(fn,sizeof(fn),"%s/asph_t%06.0fs.bin",cfg.outdir.c_str(),simt);
            FILE* f=fopen(fn,"wb");
            if (f){
                DumpHeader hh{};
                hh.magic=0x41535048u; hh.version=3;
                hh.nx=cfg.nx; hh.ny=cfg.ny; hh.nz=cfg.nz;
                hh.cx=cx; hh.cy=cy; hh.cz=cz;
                hh.sx=cfg.dump_sx; hh.sy=cfg.dump_sy; hh.sz=cfg.dump_sz;
                hh.nfields=6; hh.t_sim=(float)simt; hh.dt=dt;
                hh.lx=Lx; hh.ly=Ly; hh.lz=Lz;
                fwrite(&hh,sizeof(hh),1,f);
                fwrite(h_coarse,sizeof(float),coarseBytes/4,f);
                fclose(f); ++dumps;
            }
            printf("%-8d +%-8.0fs %-8.1f %-9.4f CFL %.2f | cloud %5.2f%% "
                   "ql_max %.3f g/kg | w_max %5.2f m/s | precip %.3f mm\n",
                   step,simt,now_s()-wall0,dt, dt_required>0?dt/dt_required:0.f,
                   cloudcov,qlmax*1e3f,wmax,mm);

            if (cfg.validate){
                double* d_tot=nullptr; CK(cudaMalloc(&d_tot,6*sizeof(double)));
                CK(cudaMemset(d_tot,0,6*sizeof(double)));
                kernel_audit<<<1024,TPB>>>(pos,vel,thq,d_tot,N);
                CK(cudaMemcpy(h_tot,d_tot,6*sizeof(double),cudaMemcpyDeviceToHost));
                CK(cudaFree(d_tot));
                printf("  AUDIT  KE %.4e  PE %.4e  IE %.4e  E_tot %.4e  "
                       "water %.4e  pz %.4e\n",
                       h_tot[0],h_tot[1],h_tot[2],h_tot[0]+h_tot[1]+h_tot[2],
                       h_tot[3],h_tot[4]);
                printf("  DRIFT  dKE %.2e  dE %.2e (%.3f%%/h)  dwater %.2e\n",
                       h_tot[0]-h_tot0[0],
                       (h_tot[0]+h_tot[1]+h_tot[2])-(h_tot0[0]+h_tot0[1]+h_tot0[2]),
                       100.0*( (h_tot[0]+h_tot[1]+h_tot[2])
                              -(h_tot0[0]+h_tot0[1]+h_tot0[2]) )
                       /fmax(1.0,fabs(h_tot0[0]+h_tot0[1]+h_tot0[2]))
                       *3600.0/fmax(1.0,simt),
                       h_tot[3]-h_tot0[3]);
            }
            toc(T_DUMP);
            next_dump += cfg.dump_every;
        }

        double wall=now_s();
        if (step % cfg.report == 0){
            double dw=wall-wall_report, ds=simt-sim_report;
            printf("  [step %d] sim-s/wall-s=%.2f  dt=%.4f  req=%.4f  sorts=%d  trips=%d | ",
                   step,dw>0?ds/dw:0.0,dt,dt_required,sort_count,trips);
            for (int k=0;k<T_COUNT;++k){
                if (tacc[k]>0) printf("%s %.2f  ",tnames[k],tacc[k]*1e3);
                tacc[k]=0;
            }
            printf("\n");
            wall_report=wall; sim_report=simt;
        }
    }

    CK(cudaStreamSynchronize(0));
    double wall=now_s()-wall0, rate=simt/wall;
    printf("\n=== DONE ==========================================================\n");
    printf("wall clock   : %.1f s\n",wall);
    printf("simulated    : %.2f sim h\n",simt/3600.0);
    printf("throughput   : %.2f sim-s per wall-s (%.2fx real time)\n",rate,rate);
    printf("24 h forecast: %.2f wall-h\n",24.0*3600.0/rate/3600.0);
    printf("parcel-steps : %.2f G/s\n",(double)N*(double)step/wall/1e9);
    printf("sorts %d, dumps %d -> %s/\n",sort_count,dumps,cfg.outdir.c_str());
    printf("CFL trips %d (tolerated %d)\n",trips,cfg.max_trip);
    printf("final dt %.4f s (required %.4f, static %.4f)\n",dt,dt_required,dt_static);
    printf("===================================================================\n");
    return 0;
}
