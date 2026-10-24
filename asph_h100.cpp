
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