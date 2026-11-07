extern "C" __global__ void make_weights(const int*y,const int*x,float*out,
    int H,int W,int R,float sigma,int nominal,int block){
    int i=blockIdx.x*blockDim.x+threadIdx.x,N=H*W,K=2*R+1;
    if(i>=N)return;
    int iy=i/W,ix=i%W;float norm=0.f;
    for(int dy=-R;dy<=R;dy++)for(int dx=-R;dx<=R;dx++){
        int jy=iy+dy,jx=ix+dx,slot=(dy+R)*K+dx+R;float w=0.f;
        if(jy>=0&&jy<H&&jx>=0&&jx<W){
            int j=jy*W+jx;
            float py=nominal?dy*block:y[j]-y[i],px=nominal?dx*block:x[j]-x[i];
            if(fabsf(py)<=3.f*sigma&&fabsf(px)<=3.f*sigma)
                w=expf(-(py*py+px*px)/(2.f*sigma*sigma));
        }
        out[slot*N+i]=w;norm+=w;
    }
    for(int slot=0;slot<K*K;slot++)out[slot*N+i]/=norm;
}

extern "C" __global__ void ssim_forward(const float*x,const float*y,const float*w,
    float*values,float*stats,int H,int W,int R){
    int z=blockIdx.x*blockDim.x+threadIdx.x,N=H*W,K=2*R+1;
    if(z>=3*N)return;
    int i=z/3,c=z%3,iy=i/W,ix=i%W;
    if(iy<R||iy>=H-R||ix<R||ix>=W-R){values[z]=0.f;return;}
    float mx=0.f,my=0.f;
    for(int dy=-R;dy<=R;dy++)for(int dx=-R;dx<=R;dx++){
        int slot=(dy+R)*K+dx+R,j=(iy+dy)*W+ix+dx;
        float a=w[slot*N+i];mx+=a*x[3*j+c];my+=a*y[3*j+c];
    }
    float vx=0.f,vy=0.f,ve=0.f;
    for(int dy=-R;dy<=R;dy++)for(int dx=-R;dx<=R;dx++){
        int slot=(dy+R)*K+dx+R,j=(iy+dy)*W+ix+dx;
        float a=w[slot*N+i],px=x[3*j+c]-mx,py=y[3*j+c]-my,e=px-py;
        vx+=a*px*px;vy+=a*py*py;ve+=a*e*e;
    }
    float dm=mx-my,lumden=mx*mx+my*my+.0001f,den=vx+vy+.0009f;
    float a=dm*dm/lumden,b=ve/den;
    values[z]=a+b-a*b;
    stats[5*z]=mx;stats[5*z+1]=my;
    stats[5*z+2]=(1.f-b)*(2.f*dm/lumden-2.f*mx*dm*dm/(lumden*lumden));
    stats[5*z+3]=2.f*(1.f-a)*(1.f-b)/den;
    stats[5*z+4]=-2.f*(1.f-a)/den;
}

extern "C" __global__ void ssim_backward(const float*x,const float*y,const float*w,
    const float*stats,const float*upstream,float*grad,int H,int W,int R){
    int z=blockIdx.x*blockDim.x+threadIdx.x,N=H*W,K=2*R+1;
    if(z>=3*N)return;
    int j=z/3,c=z%3,jy=j/W,jx=j%W;float result=0.f;
    // Gather the transpose of the overlapping-window operator. Each thread
    // owns one output derivative, so no contended scatter/atomic is needed.
    for(int dy=-R;dy<=R;dy++)for(int dx=-R;dx<=R;dx++){
        int iy=jy-dy,ix=jx-dx;
        if(iy<R||iy>=H-R||ix<R||ix>=W-R)continue;
        int i=iy*W+ix,slot=(dy+R)*K+dx+R,s=5*(3*i+c);
        float weight=w[slot*N+i];
        result+=weight*(stats[s+2]+stats[s+3]*(x[z]-stats[s])+stats[s+4]*(y[z]-stats[s+1]));
    }
    grad[z]=result*upstream[0]/(3.f*(H-2*R)*(W-2*R));
}
