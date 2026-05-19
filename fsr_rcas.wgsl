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

// NOTE: FSR_RCAS_H packs its 16-bit math across TWO horizontally-adjacent
// output pixels (its outputs are AH2 per channel). A fragment shader runs one
// pixel per invocation, so that pixel-pair packing is structurally impossible
// here — it belongs in fsr_rcas_compute.wgsl. What is portable to a fragment
// shader is keeping the per-channel math vectorized as vec3<f16> so the RGB
// lanes co-issue, instead of splitting into 9 separate scalars.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
    let FSR_RCAS_LIMIT : f16 = f16(0.25 - (1.0 / 16.0));
	// Algorithm uses minimal 3x3 pixel neighborhood.
	//    b
	//  d e f
	//    h
	// Pixel sampling stays in integer coords; loaded fp32 texels are narrowed to fp16.
    let sp = vec2<i32>(in.clip_position.xy);
	let b : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(0, -1), 0).rgb);
	let d : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(-1, 0), 0).rgb);
	let e : vec3<f16> = vec3<f16>(textureLoad(input, sp, 0).rgb);
	let f : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(1, 0), 0).rgb);
	let h : vec3<f16> = vec3<f16>(textureLoad(input, sp + vec2<i32>(0, 1), 0).rgb);

	// Luma times 2.
	var bL :f16  = b.b * 0.5 + (b.r * 0.5 + b.g);
	var dL:f16  = d.b * 0.5 + (d.r * 0.5 + d.g);
	var eL:f16  = e.b * 0.5 + (e.r * 0.5 + e.g);
	var fL:f16  = f.b * 0.5 + (f.r * 0.5 + f.g);
	var hL:f16  = h.b * 0.5 + (h.r * 0.5 + h.g);

	// Noise detection.
	var nz:f16 = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
	nz = saturate(abs(nz) * 1.0/(max3f(max3f(bL, dL, eL), fL, hL) - min3f(min3f(bL, dL, eL), fL, hL)));
	nz = -0.5 * nz + 1.0;

	// Min and max of ring (RGB packed).
	var mn4 :vec3<f16> = min(min(min(b, d), f), h);
	var mx4 :vec3<f16> = max(max(max(b, d), f), h);
	// Immediate constants for peak range.
	var peakC :vec2<f16> = vec2<f16>( 1.0, -1.0 * 4.0 );
	// Limiters, these need to be high precision RCPs.
	var hitMin :vec3<f16> = min(mn4, e) * (vec3<f16>(1.0) / (4.0 * mx4));
	var hitMax :vec3<f16> = (vec3<f16>(peakC.x) - max(mx4, e)) * (vec3<f16>(1.0) / (4.0 * mn4 + vec3<f16>(peakC.y)));
	var lobeRGB :vec3<f16> = max(-hitMin, hitMax);
	var lobe :f16 = max(-FSR_RCAS_LIMIT, min(max3f(lobeRGB.r, lobeRGB.g, lobeRGB.b), f16(0.0))) * f16(resolution.sharpness);

	// Apply noise removal.
//	lobe = lobe * nz;

	// Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
	var rcpL :f16  = 1.0/(4.0 * lobe + 1.0);
	var c:vec3<f16> = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;
	return vec4<f32>(vec3<f32>(c), 1.0);
}
