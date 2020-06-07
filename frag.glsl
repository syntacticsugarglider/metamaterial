precision mediump float;

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
uniform sampler2D sequenceTexture;

const int MAX_MARCHING_STEPS=255;
const int MAX_SWITCHES=10;
const float MIN_DIST=0.;
const float MAX_DIST=100.;
const float EPSILON=.0001;
const float PLANE_DIST=10.;
const vec3 NORMAL_VEC=vec3(-1,0,0);
const float R2=2.5;
const float R1=2.;

mat3 inverse(mat3 m){
    float a00=m[0][0],a01=m[0][1],a02=m[0][2];
    float a10=m[1][0],a11=m[1][1],a12=m[1][2];
    float a20=m[2][0],a21=m[2][1],a22=m[2][2];
    
    float b01=a22*a11-a12*a21;
    float b11=-a22*a10+a12*a20;
    float b21=a21*a10-a11*a20;
    
    float det=a00*b01+a01*b11+a02*b21;
    
    return mat3(b01,(-a22*a01+a02*a21),(a12*a01-a02*a11),
    b11,(a22*a00-a02*a20),(-a12*a00+a02*a10),
    b21,(-a21*a00+a01*a20),(a11*a00-a01*a10))/det;
}

mat3 jacobian(vec3 pos,float c){
    float r=length(pos);
    float xdx=(R2-c)/R2+c/r-(c*pow(pos.x,2.))/pow(r,3.);
    float xdy=-c*pos.x*pos.y/pow(r,3.);
    float xdz=-c*pos.x*pos.z/pow(r,3.);
    vec3 col1=vec3(xdx,xdy,xdz);
    float ydx=-c*pos.x*pos.y/pow(r,3.);
    float ydy=(R2-c)/R2+c/r-(c*pow(pos.y,2.))/pow(r,3.);
    float ydz=-c*pos.y*pos.z/pow(r,3.);
    vec3 col2=vec3(ydx,ydy,ydz);
    float zdx=-c*pos.x*pos.z/pow(r,3.);
    float zdy=-c*pos.y*pos.z/pow(r,3.);
    float zdz=(R2-c)/R2+c/r-(c*pow(pos.z,2.))/pow(r,3.);
    vec3 col3=vec3(zdx,zdy,zdz);
    return mat3(col1,col2,col3);
    
}

mat3 inv_jacobian(vec3 pos,float c){
    float r=length(pos);
    float xdx=1.+(pow(pos.x,2.)*c)/pow(r,3.)-c/r;
    float xdy=pos.x*pos.y*R1/pow(r,3.);
    float xdz=pos.z*pos.z*R1/pow(r,3.);
    vec3 col1=vec3(xdx,xdy,xdz);
    float ydx=pos.x*pos.y*R1/pow(r,3.);
    float ydy=1.+(pow(pos.y,2.)*c)/pow(r,3.)-c/r;
    float ydz=pos.z*pos.y*R1/pow(r,3.);
    vec3 col2=vec3(ydx,ydy,ydz);
    float zdx=pos.z*pos.z*R1/pow(r,3.);
    float zdy=pos.z*pos.y*R1/pow(r,3.);
    float zdz=1.+(pow(pos.z,2.)*c)/pow(r,3.)-c/r;
    vec3 col3=vec3(zdx,zdy,zdz);
    return mat3(col1,col2,col3)*R2/(R2-c);
    
}

float sdBox(vec3 p,vec3 b){
    vec3 d=abs(p)-b;
    return length(max(d,0.))
    +min(max(d.x,max(d.y,d.z)),0.);
}

float opUnion(float d1,float d2){return min(d1,d2);}

float sdSphere(vec3 p,vec3 center,float radius){
    return length(p-center)-radius;
}

float sdPlane(vec3 p,vec3 n,float h){
    // n must be normalized
    return dot(p,n)+h;
}

vec3 opRep(in vec3 p,in vec3 c)
{
    return mod(p,c)-.5*c;
}

mat3 rotateY(float theta){
    float c=cos(theta);
    float s=sin(theta);
    return mat3(
        vec3(c,0,s),
        vec3(0,1,0),
        vec3(-s,0,c)
    );
}

float distSphere(vec3 samplePoint,vec3 ray,vec3 center){
    vec3 normal=normalize(samplePoint-center);
    vec3 other=vec3(0,0,-1);
    vec3 sum=normal+other;
    mat3 R=2./dot(sum,sum)*mat3(sum.x*sum.x,sum.x*sum.y,sum.x*sum.z,sum.y*sum.x,sum.y*sum.y,sum.y*sum.z,sum.z*sum.x,sum.z*sum.y,sum.z*sum.z);
    R=R-mat3(1,0,0,0,1,0,0,0,1);
    
    float radius=length(samplePoint-center);
    
    vec3 newVec=R*ray;
    return 2.*radius*newVec.z;
    
}

float sceneSDF(vec3 samplePoint){
    float checkerboard=sdPlane(samplePoint,NORMAL_VEC,PLANE_DIST);
    
    return checkerboard;
}

float innerSDF(vec3 samplePoint){
    float cloaked=sdBox(samplePoint,vec3(1,1,1));
    return cloaked;
}

vec3 toAnnulus(vec3 init,float c){
    float l=length(init);
    return init/l*(c+(R2-c)/R2*l);
}

vec3 toSphere(vec3 init,float c){
    float l=length(init);
    return init/l*(l-c)*R2/(R2-c);
    
}

vec3 shortestDistanceToSurface(vec3 eye,vec3 marchingDirection,float start,float end,float c){
    float depth=start;
    bool enterAnnulus=false;
    bool inCenter=false;
    bool exit=false;
    float maximum=0.;
    bool finalExit=false;
    mat3 jac;
    for(int j=0;j<MAX_SWITCHES;j++){
        if((enterAnnulus||inCenter)&&!exit){
            maximum=distSphere(eye,marchingDirection,vec3(0,0,0));
        }
        
        for(int i=0;i<MAX_MARCHING_STEPS;i++){
            vec3 location=eye+depth*marchingDirection;
            float dist;
            if(exit){
                dist=-sdSphere(location,vec3(0,0,0),R2);
            }
            else if(enterAnnulus){
                float limit=(R1-c)*R2/(R2-c);
                dist=sdSphere(location,vec3(0,0,0),limit);
            }
            else if(inCenter){
                dist=innerSDF(location);
            }
            else if(finalExit){
                dist=sceneSDF(location);
            }
            else{
                dist=min(sceneSDF(location),sdSphere(location,vec3(0,0,0),R2));
            }
            
            if(dist<EPSILON){
                if(abs(length(location)-R2)<EPSILON&&(!enterAnnulus)&&(!finalExit)){
                    
                    eye=location;
                    // jac=inverse(jacobian(eye,c));
                    // marchingDirection=normalize(jac*marchingDirection);
                    depth=start;
                    enterAnnulus=true;
                    
                    inCenter=false;
                    
                    break;
                }
                else if(exit){
                    
                    eye=toAnnulus(location,c);
                    // jac=jacobian(eye,c);
                    // marchingDirection=normalize(jac*marchingDirection);
                    depth=start;
                    finalExit=true;
                    enterAnnulus=false;
                    exit=false;
                    break;
                    
                }
                else if(enterAnnulus){
                    jac=jacobian(location,c);
                    marchingDirection=normalize(jac*marchingDirection);
                    depth=start;
                    eye=toAnnulus(location,c);
                    enterAnnulus=false;
                    inCenter=true;
                    
                    break;
                }
                else{
                    return eye+depth*marchingDirection;
                }
            }
            depth+=dist;
            if(depth>=end){
                return eye+end*marchingDirection;
            }
            if(depth>=maximum&&enterAnnulus&&(!exit)){
                enterAnnulus=false;
                eye=eye+(maximum)*marchingDirection;// added padding of 0.1 so it isn't screwed when it leaves.
                // jac=jacobian(eye,c);
                // marchingDirection=normalize(jac*marchingDirection);
                break;
            }
            if(depth>=maximum&&inCenter){
                inCenter=false;
                enterAnnulus=true;
                exit=true;
                eye=eye+maximum*marchingDirection;// added padding of 0.1 so it isn't screwed when it leaves.
                jac=inverse(jacobian(eye,c));
                marchingDirection=normalize(jac*marchingDirection);
                eye=toSphere(eye,c);
                break;
            }
            
        }
    }
    return eye+end*marchingDirection;
}

vec3 estimateNormal(vec3 p){
    return normalize(vec3(
            sceneSDF(vec3(p.x+EPSILON,p.y,p.z))-sceneSDF(vec3(p.x-EPSILON,p.y,p.z)),
            sceneSDF(vec3(p.x,p.y+EPSILON,p.z))-sceneSDF(vec3(p.x,p.y-EPSILON,p.z)),
            sceneSDF(vec3(p.x,p.y,p.z+EPSILON))-sceneSDF(vec3(p.x,p.y,p.z-EPSILON))
        ));
    }
    
    vec3 phongContribForLight(vec3 k_d,vec3 k_s,float alpha,vec3 p,vec3 eye,
    vec3 lightPos,vec3 lightIntensity){
        vec3 N=estimateNormal(p);
        vec3 L=normalize(lightPos-p);
        vec3 V=normalize(eye-p);
        vec3 R=normalize(reflect(-L,N));
        
        float dotLN=dot(L,N);
        float dotRV=dot(R,V);
        
        if(dotLN<0.){
            return vec3(0.,0.,0.);
        }
        
        if(dotRV<0.){
            return lightIntensity*(k_d*dotLN);
        }
        return lightIntensity*(k_d*dotLN+k_s*pow(dotRV,alpha));
    }
    
    vec3 phongIllumination(vec3 light1Pos,vec3 k_a,vec3 k_d,vec3 k_s,float alpha,vec3 p,vec3 eye){
        const vec3 ambientLight=.1*vec3(1.,1.,1.);
        vec3 color=ambientLight*k_a;
        
        vec3 light1Intensity=vec3(.4,.4,.4);
        
        color+=phongContribForLight(k_d,k_s,alpha,p,eye,
            light1Pos,
        light1Intensity);
        return color;
    }
    
    mat3 genViewMatrix(vec3 eye,vec3 center,vec3 up){
        // Based on gluLookAt man page
        vec3 f=normalize(center-eye);
        vec3 s=normalize(cross(f,up));
        vec3 u=cross(s,f);
        return mat3(s,u,-f);
    }
    
    vec3 rayDirection(float fieldOfView,vec2 size,vec2 fragCoord){
        vec2 xy=fragCoord-size/2.;
        float z=size.y/tan(radians(fieldOfView)/2.);
        return normalize(vec3(xy,-z));
    }
    void main(){
        vec3 viewDir=rayDirection(90.,u_resolution,gl_FragCoord.xy);
        vec2 viewPerspective=vec2(u_mouse.x,1.-u_mouse.y);
        vec3 eye=vec3(7.5,(viewPerspective.y-.5)*20.,5.);
        
        eye=eye*rotateY((u_mouse.x-1.291)*3.14);
        
        mat3 viewToWorld=genViewMatrix(eye,vec3(0.,0.,0.),vec3(0.,1.,0.));
        
        vec3 worldDir=viewToWorld*viewDir;
        //float scaler=mod(u_time/2.,3.14159265);
        float c=u_time;
        
        vec3 point=shortestDistanceToSurface(eye,worldDir,MIN_DIST,MAX_DIST,c);
        if(length(point-eye)>MAX_DIST-EPSILON){
            gl_FragColor=vec4(.1,.1,.1,1.);
            return;
        }
        
        vec3 K_a;
        
        if((dot(point,NORMAL_VEC)+PLANE_DIST)<.1){
            if(mod(point.y,2.)>.05){
                if(mod(point.z,2.)>.05){
                    K_a=vec3(1,1,1);
                }
                if(mod(point.z,2.)<.05){
                    K_a=vec3(0.,0.,0.);
                }
            }
            
            if(mod(point.y,2.)<.05){
                if(mod(point.z,2.)>.05){
                    K_a=vec3(0.,0.,0.);
                }
                if(mod(point.z,2.)<.05){
                    K_a=vec3(1,1,1);
                }
            }
            vec3 K_d=K_a;
            vec3 K_s=vec3(1.,1.,1.);
            float shininess=1.;
            
            K_a=phongIllumination(vec3(-3,-10,3),K_a,K_d,K_s,shininess,point,eye);
        }
        else{
            K_a=vec3(.5,.5,.5);
            if(point.x<-.95){
                K_a.r+=.2;
            }
            if(point.x>.95){
                K_a.g-=.2;
                K_a.b-=.2;
            }
            if(point.y<-.95){
                K_a.g+=.2;
            }
            if(point.y>.95){
                K_a.r-=.2;
                K_a.b-=.2;
            }
            if(point.z<-.95){
                K_a.b+=.2;
            }
            if(point.z>.95){
                K_a.r-=.2;
                K_a.g-=.2;
            }
            
        }
        
        gl_FragColor=vec4(K_a,1.);
    }