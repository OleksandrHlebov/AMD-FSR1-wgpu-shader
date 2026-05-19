enable f16;

struct CameraUniform {
    view_proj: mat4x4<f32>,
}
struct Resolution{
   inputwidth:f32,
   inputheight:f32,
   outputwidth:f32,
   outputheight:f32,
   sharpness: f32,
}


@group(0)@binding(0)
var input: texture_2d<f32>;
@group(0)@binding(1)
var sam: sampler;
@group(1)@binding(0)// 1.
var<uniform> resolution:Resolution;
@group(2)@binding(0) // 1.
var<uniform> camera: CameraUniform;

// struct CameraUniform {
//     view_proj: mat4x4<f32>;
// };
// [[group(1),binding(0)]]// 1.
// var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
   @location(0)tex_coords: vec2<f32>,
}

fn min3f(a:f16,b:f16,c:f16)->f16{
    return min(a, min(b,c));
}
fn max3f(a:f16,b:f16,c:f16)->f16{
    return max(a, max(b,c));
}

fn saturate(num:f16)->f16{
	  return clamp(num, f16(0.0), f16(1.0));
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
	//var sp : vec2<i32> = vec2<i32>(floor(in.tex_coords * vec2<f32>(inputWidthRcas, inputHeightRcas)));
    let FSR_RCAS_LIMIT : f16 = f16(0.25 - (1.0 / 16.0));
	// Algorithm uses minimal 3x3 pixel neighborhood.
	//    b
	//  d e f
	//    h
	// Pixel sampling stays in integer coords; loaded fp32 texels are narrowed to fp16.
    let sp = vec2<i32>(in.clip_position.xy);
	let b : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(0, -1), 0).rgb);
	let d : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(-1, 0), 0).rgb);
	var e : vec3<f16> = vec3<f16>(textureLoad(input, sp, 0).rgb);
	let f : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(1, 0), 0).rgb);
	let h : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(0, 1), 0).rgb);
	// Rename (32-bit) or regroup (16-bit).
	var bR :f16 = b.r;
	var bG :f16  = b.g;
	var bB :f16  = b.b;
	var dR :f16  = d.r;
	var dG :f16  = d.g;
	var dB :f16  = d.b;
	var eR :f16  = e.r;
	var eG :f16  = e.g;
	var eB :f16  = e.b;
	var fR :f16 = f.r;
	var fG :f16  = f.g;
	var fB :f16  = f.b;
	var hR :f16 = h.r;
	var hG :f16  = h.g;
	var hB:f16  = h.b;

	var nz:f16 = f16(0.0);

	// Luma times 2.
	var bL :f16  = bB * 0.5 + (bR * 0.5 + bG);
	var dL:f16  = dB * 0.5 + (dR * 0.5 + dG);
	var eL:f16  = eB * 0.5 + (eR * 0.5 + eG);
	var fL:f16  = fB * 0.5 + (fR * 0.5 + fG);
	var hL:f16  = hB * 0.5 + (hR * 0.5 + hG);

	// Noise detection.
	nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
	nz = saturate(abs(nz) * 1.0/(max3f(max3f(bL, dL, eL), fL, hL) - min3f(min3f(bL, dL, eL), fL, hL)));
	nz = -0.5 * nz + 1.0;

	// Min and max of ring.
	var mn4R :f16 =  min(min3f(bR, dR, fR), hR);
	var mn4G :f16  = min(min3f(bG, dG, fG), hG);
	var mn4B :f16  = min(min3f(bB, dB, fB), hB);
	var mx4R :f16  = max(max3f(bR, dR, fR), hR);
	var mx4G :f16  = max(max3f(bG, dG, fG), hG);
	var mx4B :f16  = max(max3f(bB, dB, fB), hB);
	// Immediate constants for peak range.
	var peakC :vec2<f16> = vec2<f16>( 1.0, -1.0 * 4.0 );
	// Limiters, these need to be high precision RCPs.
	var hitMinR :f16  = min(mn4R, eR) * 1.0/(4.0 * mx4R);
	var hitMinG :f16 = min(mn4G, eG) * 1.0/(4.0 * mx4G);
	var hitMinB :f16  = min(mn4B, eB) * 1.0/(4.0 * mx4B);
	var hitMaxR :f16 = (peakC.x - max(mx4R, eR)) * 1.0/(4.0 * mn4R + peakC.y);
	var hitMaxG :f16  = (peakC.x - max(mx4G, eG)) * 1.0/(4.0 * mn4G + peakC.y);
	var hitMaxB :f16 = (peakC.x - max(mx4B, eB)) * 1.0/(4.0 * mn4B + peakC.y);
	var lobeR:f16  = max(-hitMinR, hitMaxR);
	var lobeG:f16  = max(-hitMinG, hitMaxG);
	var lobeB :f16  = max(-hitMinB, hitMaxB);
	var lobe :f16  = max(-FSR_RCAS_LIMIT, min(max3f(lobeR, lobeG, lobeB), f16(0.0))) * f16(resolution.sharpness);

	// Apply noise removal.
//	lobe = lobe * nz;

	// Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
	var rcpL :f16  = 1.0/(4.0 * lobe + 1.0);
	var c:vec3<f16>  = vec3<f16>(
		(lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) * rcpL,
		(lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) * rcpL,
		(lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) * rcpL
	);
	return vec4<f32>(vec3<f32>(c), 1.0);
}
