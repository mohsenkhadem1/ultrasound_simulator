// Standalone material-Gaussian renderer shared by the paired comparison views.
const vertex=`#version 300 es
precision highp float;
layout(location=0)in vec2 corner;layout(location=1)in vec3 center;layout(location=2)in vec3 ca;layout(location=3)in vec3 cb;layout(location=4)in vec4 color;
uniform vec3 eye,right,up,forward;uniform vec2 viewport;
out vec2 uv;flat out vec4 rgba;
void main(){uv=corner*3.;rgba=color;vec3 delta=center-eye;float z=dot(delta,forward),x=dot(delta,right),y=dot(delta,up);
if(z<.005){rgba.a=0.;gl_Position=vec4(2.,2.,2.,1.);return;}
mat3 cov=mat3(ca.x,ca.y,ca.z,ca.y,cb.x,cb.y,ca.z,cb.y,cb.z);float f=viewport.y*1.20710678;
vec3 jx=f/z*(right-x/z*forward),jy=f/z*(up-y/z*forward);float a=dot(jx,cov*jx)+.36,b=dot(jx,cov*jy),c=dot(jy,cov*jy)+.36;
float root=sqrt(max(0.,(a-c)*(a-c)+4.*b*b));float l1=max(.2,.5*(a+c+root)),l2=max(.2,.5*(a+c-root));
vec2 axis=abs(b)>1e-7?normalize(vec2(b,l1-a)):(a>=c?vec2(1.,0.):vec2(0.,1.));
vec2 offset=3.*(corner.x*sqrt(l1)*axis+corner.y*sqrt(l2)*vec2(-axis.y,axis.x));
gl_Position=vec4(vec2(2.*f*x/(viewport.x*z),2.*f*y/(viewport.y*z))+2.*offset/viewport,0.,1.);}`;
const fragment=`#version 300 es
precision highp float;in vec2 uv;flat in vec4 rgba;out vec4 outColor;
void main(){float q=dot(uv,uv);if(q>9.)discard;float a=rgba.a*exp(-.5*q);if(a<.0039)discard;outColor=vec4(rgba.rgb*a,a);}`;
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>a.map(v=>v/Math.hypot(...a));
export function camera({az,el,radius,target}){const eye=[target[0]+radius*Math.cos(el)*Math.cos(az),target[1]+radius*Math.cos(el)*Math.sin(az),target[2]+radius*Math.sin(el)],forward=norm(sub(target,eye)),right=norm(cross(forward,[0,0,1])),up=cross(right,forward);return{eye,forward,right,up};}
export class GaussianRenderer{
 constructor(canvas){this.canvas=canvas;const gl=this.gl=canvas.getContext('webgl2',{antialias:false,alpha:false,preserveDrawingBuffer:true});if(!gl)throw Error('WebGL2 is required');
 const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
 this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
 this.uniform={};for(const name of ['eye','right','up','forward','viewport'])this.uniform[name]=gl.getUniformLocation(this.program,name);
 this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);const q=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,q);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
 this.buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);for(const [id,n,offset]of [[1,3,0],[2,3,3],[3,3,6],[4,4,9]]){gl.enableVertexAttribArray(id);gl.vertexAttribPointer(id,n,gl.FLOAT,false,52,offset*4);gl.vertexAttribDivisor(id,1);}
 gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);gl.clearColor(.035,.052,.078,1);this.count=0;
 }
 setScene(raw,rest,appearance,cam,{fixedBrain=false,cutaway=true,cutawaySign=1,isolate=false,brainOpacity=.22}={}){
 const gl=this.gl,n=(raw.length-16)/11,entries=new Float32Array(n*13),depths=new Float32Array(n),indices=[];
 for(let i=0;i<n;i++){const j=16+i*11,k=i*9,semantic=appearance[k+4];if(isolate&&semantic<3&&semantic!==1)continue;
 const source=fixedBrain&&semantic<3&&semantic!==1?rest:raw;
 if(cutaway&&semantic===0&&source[j+1]*cutawaySign<0)continue;
 const row=indices.length,b=row*13;entries.set(source.subarray(j,j+9),b);
 let alpha=appearance[k+3];if(semantic===0)alpha*=brainOpacity;if(semantic===2)alpha*=.2;
 if(semantic===1){const brightness=Math.max(appearance[k],appearance[k+1],appearance[k+2])/.88;entries[b+9]=(fixedBrain?.12:1.)*brightness;entries[b+10]=(fixedBrain?.9:.24)*brightness;entries[b+11]=(fixedBrain?.95:.55)*brightness;alpha*=.8;}
 else{entries[b+9]=appearance[k];entries[b+10]=appearance[k+1];entries[b+11]=appearance[k+2];}
 entries[b+12]=alpha;depths[row]=entries[b]*cam.forward[0]+entries[b+1]*cam.forward[1]+entries[b+2]*cam.forward[2];indices.push(row);
 }
 indices.sort((a,b)=>depths[b]-depths[a]);const sorted=new Float32Array(indices.length*13);for(let i=0;i<indices.length;i++)sorted.set(entries.subarray(indices[i]*13,indices[i]*13+13),i*13);
 gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,sorted,gl.DYNAMIC_DRAW);this.count=indices.length;
 }
 draw(cam){const gl=this.gl,rect=this.canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,1.5),w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(this.program);gl.bindVertexArray(this.vao);for(const name of ['eye','right','up','forward'])gl.uniform3fv(this.uniform[name],cam[name]);gl.uniform2f(this.uniform.viewport,w,h);gl.drawArraysInstanced(gl.TRIANGLES,0,6,this.count);}
}
