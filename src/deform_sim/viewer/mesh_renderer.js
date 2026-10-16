// Conventional indexed triangle rendering for actual Isaac ground truth.
const vertex=`#version 300 es
precision highp float;
layout(location=0)in vec3 position;layout(location=1)in vec3 normal;
uniform vec3 eye,right,up,forward;uniform float aspect;
out vec3 world;out vec3 n;
void main(){world=position;n=normal;vec3 delta=position-eye;float z=dot(delta,forward),x=dot(delta,right),y=dot(delta,up);float near=.003,far=1.5;
gl_Position=vec4(2.41421356*x/aspect,2.41421356*y,(far+near)/(far-near)*z-2.*far*near/(far-near),z);}`;
const fragment=`#version 300 es
precision highp float;
in vec3 world;in vec3 n;uniform vec3 color,eye;uniform int semantic,cutaway;uniform float cutawaySign;
out vec4 outColor;
void main(){if(semantic==0&&cutaway==1&&world.y*cutawaySign<0.)discard;vec3 norm=normalize(n);if(!gl_FrontFacing)norm=-norm;
vec3 light=normalize(vec3(-.35,-.6,.9));float diffuse=max(0.,dot(norm,light));float fill=abs(dot(norm,normalize(vec3(.7,.1,.4))));
vec3 halfVector=normalize(light+normalize(eye-world));float spec=pow(max(0.,dot(norm,halfVector)),32.);
vec3 shaded=color*(.34+.53*diffuse+.15*fill)+.12*spec;outColor=vec4(shaded,1.);}`;
export class TriangleMeshRenderer{
 constructor(canvas){this.canvas=canvas;const gl=this.gl=canvas.getContext('webgl2',{antialias:true,alpha:false,preserveDrawingBuffer:true});if(!gl)throw Error('WebGL2 is required');
 const compile=(type,source)=>{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;};
 this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
 this.uniform={};for(const n of ['eye','right','up','forward','aspect','color','semantic','cutaway','cutawaySign'])this.uniform[n]=gl.getUniformLocation(this.program,n);
 this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);this.vertices=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.vertices);for(const [id,offset]of [[0,0],[1,3]]){gl.enableVertexAttribArray(id);gl.vertexAttribPointer(id,3,gl.FLOAT,false,24,offset*4);}
 this.indices=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);gl.clearColor(.035,.052,.078,1);this.options={};
 }
 initialize(model,indices){this.model=model;const gl=this.gl;gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);}
 setScene(packet,options){if(packet[0]!==2||packet.length!==16+this.model.vertex_count*6)throw Error('Invalid truth triangle-mesh packet');this.options=options;const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.vertices);gl.bufferData(gl.ARRAY_BUFFER,packet.subarray(16),gl.DYNAMIC_DRAW);}
 draw(cam){if(!this.model)return;const gl=this.gl,rect=this.canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,1.5),w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);gl.bindVertexArray(this.vao);
 for(const key of ['eye','right','up','forward'])gl.uniform3fv(this.uniform[key],cam[key]);gl.uniform1f(this.uniform.aspect,w/h);gl.uniform1i(this.uniform.cutaway,this.options.cutaway?1:0);gl.uniform1f(this.uniform.cutawaySign,this.options.cutawaySign??1);
 for(const group of this.model.groups){if(this.options.isolate&&group.semantic<3&&group.semantic!==1)continue;gl.uniform3fv(this.uniform.color,group.color);gl.uniform1i(this.uniform.semantic,group.semantic);gl.drawElements(gl.TRIANGLES,group.index_count,gl.UNSIGNED_INT,group.first_index*4);}
 }
}
