# AMD-FSR1-wgpu-shader
AMD FSR 1.0 implementation in wgsl for uses with webgpu 

Fixed version of the original port. Compute paths were untouched as there is no real benefit over full screen quad.
The benefit would be seen if used with workgroup local storage to share sampled pixels, avoiding read into global memory.
