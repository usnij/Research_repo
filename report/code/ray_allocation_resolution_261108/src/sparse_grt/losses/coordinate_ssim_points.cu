// Spatially indexed all-centre coordinate SSIM. No fixed neighbour cap.
// Identical pixels are compressed with exact population moments, retaining
// both neighbour and centre multiplicity; gradients remain per input sample.

__device__ __forceinline__ int lo_bin(int p, int radius, int cell) {
    return max(0, p-radius)/cell;
}
__device__ __forceinline__ int hi_bin(int p, int radius, int cell, int bins) {
    return min(bins-1, (p+radius)/cell);
}
__device__ __forceinline__ float spatial_weight(int dy, int dx, int radius, float inv_2s2) {
    if (abs(dy)>radius || abs(dx)>radius) return 0.f;
    return expf(-(float(dy)*float(dy)+float(dx)*float(dx))*inv_2s2);
}

extern "C" __global__ void aggregate_pixels(const float* p, const float* t,
    const int* order, const int* starts, float* moments, int groups) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*groups) return;
    int i=z/3,c=z%3,start=starts[i],end=starts[i+1];
    float mp=0.f,mt=0.f,inv=1.f/float(end-start);
    float p0=p[3*order[start]+c],t0=t[3*order[start]+c];
    for(int k=start;k<end;k++) {
        int j=3*order[k]+c; mp+=p[j]-p0; mt+=t[j]-t0;
    }
    mp=p0+mp*inv;mt=t0+mt*inv;
    float vp=0.f,vt=0.f,ve=0.f;
    for(int k=start;k<end;k++) {
        int j=3*order[k]+c;
        float dp=p[j]-mp,dt=t[j]-mt,de=dp-dt;
        vp+=dp*dp;vt+=dt*dt;ve+=de*de;
    }
    int s=5*z;
    moments[s]=mp;moments[s+1]=mt;
    moments[s+2]=vp*inv;moments[s+3]=vt*inv;moments[s+4]=ve*inv;
}

extern "C" __global__ void point_ssim_forward(const float* moments,
    const int* starts,const int* y,const int* x,const int* group_order,
    const int* offsets,float* values,float* stats,float* inv_norm,
    int groups,int cell,int bh,int bw,int radius,float sigma) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*groups) return;
    int i=z/3,c=z%3,iy=y[i],ix=x[i];
    int ly=lo_bin(iy,radius,cell),hy=hi_bin(iy,radius,cell,bh);
    int lx=lo_bin(ix,radius,cell),hx=hi_bin(ix,radius,cell,bw);
    float norm=0.f,mp=0.f,mt=0.f,inv_2s2=1.f/(2.f*sigma*sigma);
    for(int by=ly;by<=hy;by++) for(int bx=lx;bx<=hx;bx++) {
        int bin=by*bw+bx;
        for(int k=offsets[bin];k<offsets[bin+1];k++) {
            int j=group_order[k],s=5*(3*j+c);
            float w=spatial_weight(y[j]-iy,x[j]-ix,radius,inv_2s2)*float(starts[j+1]-starts[j]);
            norm+=w;mp+=w*moments[s];mt+=w*moments[s+1];
        }
    }
    float inv=1.f/norm;mp*=inv;mt*=inv;
    if(c==0) inv_norm[i]=inv;
    float vp=0.f,vt=0.f,ve=0.f;
    for(int by=ly;by<=hy;by++) for(int bx=lx;bx<=hx;bx++) {
        int bin=by*bw+bx;
        for(int k=offsets[bin];k<offsets[bin+1];k++) {
            int j=group_order[k],s=5*(3*j+c);
            float w=spatial_weight(y[j]-iy,x[j]-ix,radius,inv_2s2)*float(starts[j+1]-starts[j])*inv;
            float dp=moments[s]-mp,dt=moments[s+1]-mt,de=dp-dt;
            vp+=w*(moments[s+2]+dp*dp);
            vt+=w*(moments[s+3]+dt*dt);
            ve+=w*(moments[s+4]+de*de);
        }
    }
    float dm=mp-mt,ld=mp*mp+mt*mt+.0001f,den=vp+vt+.0009f;
    float a=dm*dm/ld,b=ve/den;
    values[z]=(a+b-a*b)*float(starts[i+1]-starts[i]);
    int s=5*z;
    stats[s]=mp;stats[s+1]=mt;
    stats[s+2]=(1.f-b)*(2.f*dm/ld-2.f*mp*dm*dm/(ld*ld));
    stats[s+3]=2.f*(1.f-a)*(1.f-b)/den;
    stats[s+4]=-2.f*(1.f-a)/den;
}

extern "C" __global__ void point_ssim_backward(const float* p,const float* t,
    const int* y,const int* x,const int* starts,const int* uy,const int* ux,
    const int* group_order,const int* offsets,const float* stats,const float* inv_norm,
    const float* upstream,float* grad,int n,int cell,int bh,int bw,int radius,float sigma) {
    int z=blockIdx.x*blockDim.x+threadIdx.x;
    if(z>=3*n) return;
    int j=z/3,c=z%3,jy=y[j],jx=x[j];
    int ly=lo_bin(jy,radius,cell),hy=hi_bin(jy,radius,cell,bh);
    int lx=lo_bin(jx,radius,cell),hx=hi_bin(jx,radius,cell,bw);
    float result=0.f,inv_2s2=1.f/(2.f*sigma*sigma);
    // Every thread owns one original derivative. Contributions from repeated
    // centres are multiplied by their counts; no atomic gradient scatter.
    for(int by=ly;by<=hy;by++) for(int bx=lx;bx<=hx;bx++) {
        int bin=by*bw+bx;
        for(int k=offsets[bin];k<offsets[bin+1];k++) {
            int i=group_order[k],s=5*(3*i+c);
            float w=spatial_weight(uy[i]-jy,ux[i]-jx,radius,inv_2s2)*inv_norm[i]
                    *float(starts[i+1]-starts[i]);
            result+=w*(stats[s+2]+stats[s+3]*(p[z]-stats[s])+stats[s+4]*(t[z]-stats[s+1]));
        }
    }
    grad[z]=result*upstream[0]/(3.f*float(n));
}
