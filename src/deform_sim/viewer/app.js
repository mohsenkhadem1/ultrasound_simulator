const $=id=>document.getElementById(id), canvas=$('scene');
const gl=canvas.getContext('webgl2',{antialias:false,alpha:false,preserveDrawingBuffer:true});
if(!gl)throw new Error('WebGL2 is required for anisotropic Gaussian splatting.');
const debug=gl.getExtension('WEBGL_debug_renderer_info');
const renderer=debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
$('renderer').textContent=renderer;
const vertex=`#version 300 es
precision highp float;
layout(location=0)in vec2 corner;
layout(location=1)in vec3 center;
layout(location=2)in vec3 covA;
layout(location=3)in vec3 covB;
layout(location=4)in vec4 color;
layout(location=5)in vec2 material;
layout(location=6)in vec2 metrics;
layout(location=7)in vec3 rest;
uniform vec3 eye,right,up,forward;
uniform vec2 viewport;
uniform float focal,brainOpacity,magnify;
uniform int cutaway,tumorOnly,showPlane,colourMode;
out vec2 uv;flat out vec4 rgba;
vec3 ramp(float t){t=clamp(t,0.,1.);return mix(mix(vec3(.10,.35,.92),vec3(.1,.85,.75),smoothstep(0.,.5,t)),vec3(1.,.33,.16),smoothstep(.5,1.,t));}
void main(){
 float semantic=material.x;vec3 pos=center;
 if(semantic<3.)pos=rest+(center-rest)*magnify;
 rgba=color;uv=corner*3.;
 if((semantic<.5&&cutaway==1&&pos.y<0.)||(tumorOnly==1&&semantic<3.&&abs(semantic-1.)>.1)||(semantic>3.5&&showPlane==0)){
  rgba.a=0.;gl_Position=vec4(2.,2.,2.,1.);return;}
 if(semantic<.5)rgba.a*=brainOpacity;
 if(semantic<3.&&colourMode>0){float value=colourMode==1?metrics.x/.005:(colourMode==2?(material.y-2000.)/2000.:metrics.y/1000.);rgba.rgb=ramp(value);}
 vec3 delta=pos-eye;float z=dot(delta,forward),x=dot(delta,right),y=dot(delta,up);
 if(z<.005){rgba.a=0.;gl_Position=vec4(2.,2.,2.,1.);return;}
 mat3 cov=mat3(covA.x,covA.y,covA.z,covA.y,covB.x,covB.y,covA.z,covB.y,covB.z);
 float fx=viewport.x*focal*.5,fy=viewport.y*focal*.5;
 // Focal is vertical; account for aspect in the horizontal projection.
 fx=fy;
 vec3 jx=fx/z*(right-x/z*forward),jy=fy/z*(up-y/z*forward);
 float a=dot(jx,cov*jx)+.36,b=dot(jx,cov*jy),c=dot(jy,cov*jy)+.36;
 float root=sqrt(max(0.,(a-c)*(a-c)+4.*b*b));
 float l1=max(.2,.5*(a+c+root)),l2=max(.2,.5*(a+c-root));
 vec2 axis=abs(b)>1e-7?normalize(vec2(b,l1-a)):(a>=c?vec2(1.,0.):vec2(0.,1.));
 vec2 offset=3.*(corner.x*sqrt(l1)*axis+corner.y*sqrt(l2)*vec2(-axis.y,axis.x));
 vec2 ndc=vec2(2.*fx*x/(viewport.x*z),2.*fy*y/(viewport.y*z))+2.*offset/viewport;
 gl_Position=vec4(ndc,0.,1.);
}`;
const fragment=`#version 300 es
precision highp float;in vec2 uv;flat in vec4 rgba;out vec4 outColor;
void main(){float q=dot(uv,uv);if(q>9.)discard;float alpha=rgba.a*exp(-.5*q);if(alpha<.0039)discard;outColor=vec4(rgba.rgb*alpha,alpha);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
gl.useProgram(program);const U={};for(const n of ['eye','right','up','forward','viewport','focal','brainOpacity','magnify','cutaway','tumorOnly','showPlane','colourMode'])U[n]=gl.getUniformLocation(program,n);
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const quad=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,quad);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
const instances=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,instances);
for(const [id,size,offset] of [[1,3,0],[2,3,3],[3,3,6],[4,4,9],[5,2,13],[6,2,15],[7,3,17]]){gl.enableVertexAttribArray(id);gl.vertexAttribPointer(id,size,gl.FLOAT,false,80,offset*4);gl.vertexAttribDivisor(id,1);}
gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);gl.clearColor(.035,.055,.085,1.);
const state={count:0,static:null,raw:null,latest:null,sorted:null,indices:[],dirty:true,lastReceive:0,received:0,auto:false,paused:false,seq:-1,physicsLatest:null,estimateLatest:null,estimateMeta:null,pressDemo:false};
let azimuth=-1.02,elevation=.48,distance=.36,target=[-.006,.025,.078],cameraDirty=true;
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const l=Math.hypot(...a);return a.map(v=>v/l);};
function camera(){const eye=[target[0]+distance*Math.cos(elevation)*Math.cos(azimuth),target[1]+distance*Math.cos(elevation)*Math.sin(azimuth),target[2]+distance*Math.sin(elevation)];const forward=norm(sub(target,eye)),right=norm(cross(forward,[0,0,1])),up=cross(right,forward);return{eye,forward,right,up};}
let ws;
function send(data){if(ws?.readyState===1)ws.send(JSON.stringify(data));else fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
function toggleAuto(value){state.auto=value;$('auto').textContent=value?'Stop automatic scan':'Start automatic scan';state.pressDemo=false;$('us-press').textContent='Start fixed-plane press · US tracking';send({auto:value,us_press:false});}
function togglePause(value){state.paused=value;$('pause').textContent=value?'Resume':'Pause';send({paused:value});}
for(const [id,key,suffix] of [['press','press_mm',' mm'],['scan','y_mm',' mm'],['side','x_mm',' mm'],['yaw','yaw_deg','°']]){$(id).oninput=()=>{$(id+'-value').textContent=Number($(id).value).toFixed(1)+suffix;toggleAuto(false);send({[key]:Number($(id).value)});};}
$('geometry-source').oninput=()=>{const source=$('geometry-source').value;send({stream:source});state.latest=source==='estimate'?state.estimateLatest:(state.physicsLatest?.length>16?state.physicsLatest:null);cameraDirty=true;};
$('us-press').onclick=()=>{state.pressDemo=!state.pressDemo;state.auto=false;$('auto').textContent='Start automatic scan';$('us-press').textContent=state.pressDemo?'Stop fixed-plane press':'Start fixed-plane press · US tracking';send({auto:false,us_press:state.pressDemo});};
$('auto').onclick=()=>toggleAuto(!state.auto);$('pause').onclick=()=>togglePause(!state.paused);
$('reset').onclick=()=>{state.auto=false;state.paused=false;for(const id of ['press','scan','side','yaw']){$(id).value=0;$(id+'-value').textContent='0.0'+(id==='yaw'?'°':' mm');}$('auto').textContent='Start automatic scan';$('pause').textContent='Pause';state.pressDemo=false;$('us-press').textContent='Start fixed-plane press · US tracking';send({reset:true});};
for(const id of ['opacity','cutaway','tumor-only','plane','colour','magnify'])$(id).oninput=()=>{cameraDirty=true;};
$('colour').oninput=()=>{cameraDirty=true;const mode=Number($('colour').value);$('anatomy-legend').hidden=mode!==0;$('field-legend').hidden=mode===0;$('field-low').textContent=mode===2?'2 kPa':'0';$('field-high').textContent=mode===1?'5 mm':mode===2?'4 kPa':'1 kPa';};
let dragging=false,lastX=0,lastY=0;canvas.onpointerdown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>dragging=false;canvas.onpointermove=e=>{if(!dragging)return;azimuth-=(e.clientX-lastX)*.006;elevation=Math.max(-1.25,Math.min(1.3,elevation+(e.clientY-lastY)*.005));lastX=e.clientX;lastY=e.clientY;cameraDirty=true;};canvas.onwheel=e=>{e.preventDefault();distance=Math.min(.9,Math.max(.13,distance*Math.exp(e.deltaY*.001)));cameraDirty=true;};
window.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;const map={ArrowLeft:['side',-.5],ArrowRight:['side',.5],ArrowUp:['scan',1],ArrowDown:['scan',-1],r:['press',.2],f:['press',-.2]};if(e.code==='Space'){e.preventDefault();togglePause(!state.paused);}if(map[e.key]){e.preventDefault();const[id,delta]=map[e.key];$(id).value=Number($(id).value)+delta;$(id).dispatchEvent(new Event('input'));}});
async function initialize(){const status=await(await fetch('/api/status',{cache:'no-store'})).json();$('geometry-source').value=status.ultrasound_enabled?'estimate':'simulation';$('geometry-source').options[0].disabled=!status.ultrasound_enabled;state.count=status.gaussian_count;state.static=new Float32Array(await(await fetch('/model.bin',{cache:'no-store'})).arrayBuffer());state.indices=Array.from({length:state.count},(_,i)=>i);state.sorted=new Float32Array(state.count*20);$('count').textContent=state.count.toLocaleString();state.auto=status.controls.auto;state.paused=status.controls.paused;state.pressDemo=!!status.controls.us_press;$('us-press').textContent=state.pressDemo?'Stop fixed-plane press':'Start fixed-plane press · US tracking';for(const [id,key,suffix] of [['press','press_mm',' mm'],['scan','y_mm',' mm'],['side','x_mm',' mm'],['yaw','yaw_deg','°']]){$(id).value=status.controls[key];$(id+'-value').textContent=Number(status.controls[key]).toFixed(1)+suffix;}$('pause').textContent=state.paused?'Resume':'Pause';$('auto').textContent=state.auto?'Stop automatic scan':'Start automatic scan';
 ws=new WebSocket(`ws://${location.host}/ws?stream=${$('geometry-source').value}`);ws.binaryType='arraybuffer';ws.onopen=()=>{$('connection').textContent='● Live physics';};ws.onclose=()=>{$('connection').textContent='Disconnected';$('state').textContent='Solver disconnected. The displayed state is frozen.';};ws.onerror=()=>{$('connection').textContent='Connection error';};ws.onmessage=e=>{if(typeof e.data==='string'){const m=JSON.parse(e.data);if(m.type==='error')console.error(m.message);if(m.type==='physics_status'){state.physicsLatest=new Float32Array(m.header);state.lastReceive=performance.now();state.received++;}return;}state.physicsLatest=new Float32Array(e.data);if($('geometry-source').value==='simulation')state.latest=state.physicsLatest;state.lastReceive=performance.now();state.received++;};}
function updateBuffer(cam){const n=state.count,raw=state.raw,s=state.static,sorted=state.sorted,magnification=Number($('magnify').value);const depths=new Float32Array(n);for(let i=0;i<n;i++){const j=16+i*11;const k=i*9,m=s[k+4]<3?magnification:1;depths[i]=(s[k+6]+(raw[j]-s[k+6])*m)*cam.forward[0]+(s[k+7]+(raw[j+1]-s[k+7])*m)*cam.forward[1]+(s[k+8]+(raw[j+2]-s[k+8])*m)*cam.forward[2];}state.indices.sort((a,b)=>depths[b]-depths[a]);for(let k=0;k<n;k++){const i=state.indices[k],j=16+i*11,t=i*9,b=k*20;sorted.set(raw.subarray(j,j+9),b);sorted.set(s.subarray(t,t+6),b+9);sorted[b+15]=raw[j+9];sorted[b+16]=raw[j+10];sorted.set(s.subarray(t+6,t+9),b+17);}gl.bindBuffer(gl.ARRAY_BUFFER,instances);gl.bufferData(gl.ARRAY_BUFFER,sorted,gl.DYNAMIC_DRAW);}
let frameCount=0,lastStats=performance.now(),renderFPS=0;
let lastDraw=-1e9;
function draw(now){requestAnimationFrame(draw);if(now-lastDraw<1000/60)return;lastDraw=now;const ratio=Math.min(devicePixelRatio,1.5),width=Math.round(innerWidth*ratio),height=Math.round(innerHeight*ratio);if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;gl.viewport(0,0,width,height);cameraDirty=true;}
 if(state.latest){state.raw=state.latest;state.latest=null;state.dirty=true;state.seq=state.raw[2];const physics=state.physicsLatest||state.raw;state.paused=physics[13]>.5;$('pause').textContent=state.paused?'Resume':'Pause';$('connection').textContent=state.raw[14]>.5?'Paused · invalid tissue':state.paused?'Paused':'● Live physics';if(state.raw[14]>.5){state.auto=false;$('auto').textContent='Start automatic scan';}$('motion').textContent=(state.raw[11]*1000).toFixed(2)+' mm';$('stream-rate').textContent=(state.physicsLatest||state.raw)[9].toFixed(1)+' Hz';$('motion-caption').textContent=$('geometry-source').value==='estimate'?'peak US-inferred motion':'peak reference motion';$('time').textContent=state.raw[1].toFixed(2)+' s';$('state').textContent=state.raw[14]>.5?'Invalid tissue state: physics paused. Reset the tissue to continue.':`Live Isaac GPU contact · volume J min ${state.raw[10].toFixed(3)} · ${Number($('magnify').value)===1?'actual scale':'motion magnified for viewing'}`;}
 if(state.raw&&$('geometry-source').value==='estimate'){$('connection').textContent=state.paused?'Paused':state.estimateMeta?.observable?'● US-based estimate':'US tracking · prior / last valid';$('state').textContent=`US image-driven elastic inverse · frame ${state.seq} · geometry starts from MRI prior · motion display ${$('magnify').value}×`;}
 gl.clear(gl.COLOR_BUFFER_BIT);if(!state.raw||!state.static)return;const cam=camera();if(state.dirty||cameraDirty){updateBuffer(cam);state.dirty=false;cameraDirty=false;}
 gl.useProgram(program);gl.bindVertexArray(vao);for(const key of ['eye','right','up','forward'])gl.uniform3fv(U[key],cam[key]);gl.uniform2f(U.viewport,width,height);gl.uniform1f(U.focal,1/Math.tan(45*Math.PI/360));gl.uniform1f(U.brainOpacity,Number($('opacity').value));gl.uniform1f(U.magnify,Number($('magnify').value));gl.uniform1i(U.cutaway,$('cutaway').checked?1:0);gl.uniform1i(U.tumorOnly,$('tumor-only').checked?1:0);gl.uniform1i(U.showPlane,$('plane').checked?1:0);gl.uniform1i(U.colourMode,Number($('colour').value));gl.drawArraysInstanced(gl.TRIANGLES,0,6,state.count);
 frameCount++;if(now-lastStats>1000){renderFPS=frameCount*1000/(now-lastStats);$('render-rate').textContent=renderFPS.toFixed(0);frameCount=0;lastStats=now;}window.viewerStats={received:state.received,renderFPS,streamHZ:(state.physicsLatest||state.raw)[9],inferenceHZ:state.estimateMeta?(state.estimateMeta.update_hz??1000/state.estimateMeta.inference_ms):0,sequence:state.seq,simulationTime:state.raw[1],ageMs:Math.max(0,performance.now()-state.lastReceive),renderer,count:state.count,geometrySource:$('geometry-source').value,estimatedFrame:state.estimateMeta?.frame,glError:gl.getError()};}
initialize().catch(e=>{$('connection').textContent='Viewer error';$('state').textContent=e.message;console.error(e);});requestAnimationFrame(draw);

let usFrame=-1,estimateFrame=-1;
async function ultrasoundJSON(name){const r=await fetch('/ultrasound/'+name,{cache:'no-store'});return r.ok?r.json():null;}
async function pollUltrasound(){try{const [image,estimate]=await Promise.all([ultrasoundJSON('image.json'),ultrasoundJSON('estimate.json')]);const useEstimate=$('geometry-source').value==='estimate';if(estimate&&estimate.frame!==estimateFrame){const r=await fetch('/ultrasound/'+estimate.estimate_file,{cache:'no-store'});if(r.ok){const packet=new Float32Array(await r.arrayBuffer());if(packet[2]===estimate.frame){state.estimateLatest=packet;state.estimateMeta=estimate;estimateFrame=estimate.frame;if(useEstimate)state.latest=packet;}}}const displayed=useEstimate&&state.estimateMeta?.anchor_time!==undefined?state.estimateMeta:image;if(displayed){const key=(useEstimate?'e':'p')+displayed.frame;if(usFrame!==key){$('us-image').src='/ultrasound/'+displayed.image;usFrame=key;}const age=Math.max(0,Date.now()/1000-displayed.capture_wall_time);$('us-model').textContent=`Live PSF ultrasound preview · t=${displayed.simulation_time.toFixed(2)} s · image age ${age.toFixed(1)} s`;if(estimate){$('us-estimate').textContent=`US inverse ${(estimate.update_hz??1000/Math.max(1,estimate.inference_ms)).toFixed(1)} Hz · `+(estimate.observable?`incremental motion ${estimate.max_inferred_motion_mm.toFixed(2)} mm; image fit ${estimate.image_constraint_rms_mm.toFixed(2)} mm. Out-of-plane motion follows the prior.`:estimate.message);}window.ultrasoundStats={geometrySource:$('geometry-source').value,imageFrame:displayed.frame,estimatedFrame:estimate?.frame,imageTime:displayed.simulation_time,ageSeconds:age,observable:estimate?.observable,imageModel:'coherent IQ PSF preview; not RF/SIMUS'};}}catch(e){console.warn('Ultrasound stream:',e.message);}finally{setTimeout(pollUltrasound,200);}}
pollUltrasound();
