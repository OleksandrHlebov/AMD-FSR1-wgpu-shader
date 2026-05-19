enable f16;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}
struct Resolution{
   inputwidth:f32,
   inputheight:f32,
   outputwidth:f32,
   outputheight:f32,
}

@group(0) @binding(0)
var input: texture_2d<f32>;
@group(0) @binding(1)
var sam: sampler;

@group(1) @binding(0)
var<uniform> resolution:Resolution;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Packed (FSR_EASU_H) data: every vec2<f16> holds TWO lanes of work, so the
// math below is the genuine 16-bit dual-issue path, not the scalar FSR_EASU_F
// reference retyped to f16.
struct FsrSet{
    dirPX:vec2<f16>, // dir.x for the two packed quadrants.
    dirPY:vec2<f16>, // dir.y for the two packed quadrants.
    lenP:vec2<f16>,  // len for the two packed quadrants.
}
struct FsrTap{
    aCR:vec2<f16>, // Accumulated red for the two packed taps.
    aCG:vec2<f16>,
    aCB:vec2<f16>,
    aW:vec2<f16>,  // Accumulated weight for the two packed taps.
}

// Accumulate direction and length — runs 2 quadrants in parallel.
fn FsrEasuSetH(
    fsr:FsrSet,
    pp:vec2<f16>,
    biST:bool, biUV:bool,
    lA:vec2<f16>, lB:vec2<f16>, lC:vec2<f16>, lD:vec2<f16>, lE:vec2<f16>) -> FsrSet {
    var fsr1 : FsrSet = fsr;
    // Bilinear weight for the two packed quadrants: {wS,wT} or {wU,wV}.
    var w : vec2<f16> = vec2<f16>(0.0);
    if (biST) { w = (vec2<f16>(1.0, 0.0) + vec2<f16>(-pp.x, pp.x)) * (f16(1.0) - pp.y); }
    if (biUV) { w = (vec2<f16>(1.0, 0.0) + vec2<f16>(-pp.x, pp.x)) * pp.y; }
    // Direction is the '+' diff; length converts gradient reversal to 0.
    var dc : vec2<f16> = lD - lC;
    var cb : vec2<f16> = lC - lB;
    var lenX : vec2<f16> = max(abs(dc), abs(cb));
    lenX = vec2<f16>(1.0) / lenX;
    var dirX : vec2<f16> = lD - lB;
    fsr1.dirPX = fsr1.dirPX + dirX * w;
    lenX = clamp(abs(dirX) * lenX, vec2<f16>(0.0), vec2<f16>(1.0));
    lenX = lenX * lenX;
    fsr1.lenP = fsr1.lenP + lenX * w;
    // Repeat for the y axis.
    var ec : vec2<f16> = lE - lC;
    var ca : vec2<f16> = lC - lA;
    var lenY : vec2<f16> = max(abs(ec), abs(ca));
    lenY = vec2<f16>(1.0) / lenY;
    var dirY : vec2<f16> = lE - lA;
    fsr1.dirPY = fsr1.dirPY + dirY * w;
    lenY = clamp(abs(dirY) * lenY, vec2<f16>(0.0), vec2<f16>(1.0));
    lenY = lenY * lenY;
    fsr1.lenP = fsr1.lenP + lenY * w;
    return fsr1;
}

// Filtering for a given tap — runs 2 taps in parallel.
fn FsrEasuTapH(
    fsr:FsrTap,
    offX:vec2<f16>, // Pixel x offset of the two taps.
    offY:vec2<f16>, // Pixel y offset of the two taps.
    dir:vec2<f16>,  // Gradient direction.
    len:vec2<f16>,  // Anisotropic length.
    lob:f16,        // Negative lobe strength.
    clp:f16,        // Clipping point.
    cR:vec2<f16>, cG:vec2<f16>, cB:vec2<f16> // Tap colors for the two taps.
)-> FsrTap {
    var fsr1 : FsrTap = fsr;
    // Rotate offset by direction.
    var vX : vec2<f16> = offX * dir.xx + offY * dir.yy;
    var vY : vec2<f16> = offX * (-dir.yy) + offY * dir.xx;
    // Anisotropy.
    vX = vX * len.x;
    vY = vY * len.y;
    // Compute distance^2, limited to the window.
    var d2 : vec2<f16> = vX * vX + vY * vY;
    d2 = min(d2, vec2<f16>(clp));
    // Approximation of lanczos2 (see FSR_EASU_F reference for the derivation).
    var wB : vec2<f16> = vec2<f16>(2.0 / 5.0) * d2 + vec2<f16>(-1.0);
    var wA : vec2<f16> = vec2<f16>(lob) * d2 + vec2<f16>(-1.0);
    wB = wB * wB;
    wA = wA * wA;
    wB = vec2<f16>(25.0 / 16.0) * wB + vec2<f16>(-(25.0 / 16.0 - 1.0));
    var w : vec2<f16> = wB * wA;
    // Do weighted average.
    fsr1.aCR = fsr1.aCR + cR * w;
    fsr1.aCG = fsr1.aCG + cG * w;
    fsr1.aCB = fsr1.aCB + cB * w;
    fsr1.aW = fsr1.aW + w;
    return fsr1;
}

fn gather_red_components(c: vec2<f32>) -> vec4<f16> {
   return vec4<f16>(textureGather(0,input,sam,c));
}
fn gather_green_components(c: vec2<f32>) -> vec4<f16> {
   return vec4<f16>(textureGather(1,input,sam,c));
}
fn gather_blue_components(c: vec2<f32>) -> vec4<f16> {
   return vec4<f16>(textureGather(2,input,sam,c));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
	var inputSize : vec2<f32>;
    inputSize.x = resolution.inputwidth;
    inputSize.y = resolution.inputheight;
	var outputSize : vec2<f32>;
    outputSize.x = resolution.outputwidth;
    outputSize.y = resolution.outputheight;

	//------------------------------------------------------------------------------------------------------------------------------
	  // Get position of 'f'.
	  // Pixel-coordinate math stays in fp32: at high resolutions these values exceed the
	  // precision (and range) of fp16, so floor()/normalization must be done at full precision.
	var pp:vec2<f32> = (floor(in.tex_coords * outputSize) + 0.5) / outputSize * inputSize - 0.5;
	var fp : vec2<f32> = floor(pp);
	pp = pp - fp;
	// The fractional offset is in [0,1) and feeds the kernel math, so it is safe in fp16.
	var pph : vec2<f16> = vec2<f16>(pp);
	//------------------------------------------------------------------------------------------------------------------------------
	  // 12-tap kernel.
	  //    b c
	  //  e f g h
	  //  i j k l
	  //    n o
	  // Gather 4 ordering.
	  //  a b
	  //  r g
	  // For packed FP16, need either {rg} or {ab} so using the following setup for gather in all versions,
	  //    a b    <- unused (z)
	  //    r g
	  //  a b a b
	  //  r g r g
	  //    a b
	  //    r g    <- unused (z)
	  // Allowing dead-code removal to remove the 'z's
	var p0 :vec2<f32> = fp + vec2<f32>(1.0, -1.0);
	// These are from p0 to avoid pulling two constants on pre-Navi hardware.
	var p1 : vec2<f32> = p0 + vec2<f32>(-1.0, 2.0);
	var p2: vec2<f32> = p0 + vec2<f32>(1.0, 2.0);
	var p3: vec2<f32> = p0 + vec2<f32>(0.0, 4.0);

	p0 = p0 / inputSize;
	p1 = p1 / inputSize;
	p2 = p2 / inputSize;
	p3 = p3 / inputSize;


	var bczzR : vec4<f16> = gather_red_components(p0);
	var bczzG : vec4<f16> = gather_green_components(p0);
	var bczzB : vec4<f16> = gather_blue_components(p0);
	var ijfeR : vec4<f16> = gather_red_components(p1);
	var ijfeG : vec4<f16> = gather_green_components(p1);
	var ijfeB : vec4<f16> = gather_blue_components(p1);
	var klhgR : vec4<f16> = gather_red_components(p2);
	var klhgG : vec4<f16> = gather_green_components(p2);
	var klhgB : vec4<f16> = gather_blue_components(p2);
	var zzonR : vec4<f16> = gather_red_components(p3);
	var zzonG : vec4<f16> = gather_green_components(p3);
	var zzonB : vec4<f16> = gather_blue_components(p3);
	//------------------------------------------------------------------------------------------------------------------------------
	  // Simplest multi-channel approximate luma possible (luma times 2, in 2 FMA/MAD).
	var bczzL : vec4<f16> = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
	var ijfeL : vec4<f16> = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
	var klhgL : vec4<f16> = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
	var zzonL : vec4<f16> = zzonB * 0.5 + (zzonR * 0.5 + zzonG);
	// Rename.
	var bL :f16 = bczzL.x;
	var cL :f16 = bczzL.y;
	var iL :f16 = ijfeL.x;
	var jL :f16 = ijfeL.y;
	var fL :f16 = ijfeL.z;
	var eL :f16 = ijfeL.w;
	var kL :f16 = klhgL.x;
	var lL :f16 = klhgL.y;
	var hL :f16 = klhgL.z;
	var gL :f16 = klhgL.w;
	var oL :f16 = zzonL.z;
	var nL :f16 = zzonL.w;
	// Accumulate for bilinear interpolation — 4 quadrants as 2 packed pairs.
    var fsr:FsrSet;
	fsr.dirPX = vec2<f16>(0.0, 0.0);
	fsr.dirPY = vec2<f16>(0.0, 0.0);
	fsr.lenP = vec2<f16>(0.0, 0.0);
	fsr = FsrEasuSetH(fsr, pph, true, false,
		vec2<f16>(bL, cL), vec2<f16>(eL, fL), vec2<f16>(fL, gL), vec2<f16>(gL, hL), vec2<f16>(jL, kL));
	fsr = FsrEasuSetH(fsr, pph, false, true,
		vec2<f16>(fL, gL), vec2<f16>(iL, jL), vec2<f16>(jL, kL), vec2<f16>(kL, lL), vec2<f16>(nL, oL));
	// Reduce the packed pairs to scalar dir/len.
	var dir : vec2<f16> = vec2<f16>(fsr.dirPX.x + fsr.dirPX.y, fsr.dirPY.x + fsr.dirPY.y);
	var len : f16 = fsr.lenP.x + fsr.lenP.y;
	//------------------------------------------------------------------------------------------------------------------------------
	  // Normalize with approximation, and cleanup close to zero.
	var dir2 :vec2<f16> = dir * dir;
	var dirR :f16 = dir2.x + dir2.y;
	var zro : bool = dirR < f16(1.0 / 32768.0);
	dirR = f16(1.0) / sqrt(dirR);
	if (zro) {dirR = f16(1.0);};
	if (zro) {dir.x = f16(1.0);};
	dir = dir * dirR;
	// Transform from {0 to 2} to {0 to 1} range, and shape with square.
	len = len * 0.5;
	len = len * len;
	// Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
	var stretch :f16 = (dir.x * dir.x + dir.y * dir.y) * 1.0/(max(abs(dir.x), abs(dir.y)));
	// Anisotropic length after rotation,
	//  x := 1.0 lerp to 'stretch' on edges
	//  y := 1.0 lerp to 2x on edges
	var len2:vec2<f16> = vec2<f16>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len );
	// Based on the amount of 'edge',
	// the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
	var lob : f16 = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
	// Set distance^2 clipping point to the end of the adjustable window.
	var clp : f16 = 1.0/lob;
	//------------------------------------------------------------------------------------------------------------------------------
	  // Min/max of the 4 nearest, packed: each vec2 carries {-min, max} for one channel.
	  //    b c
	  //  e f g h
	  //  i j k l
	  //    n o
	var bothR : vec2<f16> = max(max(vec2<f16>(-ijfeR.z, ijfeR.z), vec2<f16>(-klhgR.w, klhgR.w)),
		max(vec2<f16>(-ijfeR.y, ijfeR.y), vec2<f16>(-klhgR.x, klhgR.x)));
	var bothG : vec2<f16> = max(max(vec2<f16>(-ijfeG.z, ijfeG.z), vec2<f16>(-klhgG.w, klhgG.w)),
		max(vec2<f16>(-ijfeG.y, ijfeG.y), vec2<f16>(-klhgG.x, klhgG.x)));
	var bothB : vec2<f16> = max(max(vec2<f16>(-ijfeB.z, ijfeB.z), vec2<f16>(-klhgB.w, klhgB.w)),
		max(vec2<f16>(-ijfeB.y, ijfeB.y), vec2<f16>(-klhgB.x, klhgB.x)));
	// Accumulation — 12 taps as 6 packed pairs.
	var fsr2:FsrTap;
	fsr2.aCR = vec2<f16>(0.0, 0.0);
	fsr2.aCG = vec2<f16>(0.0, 0.0);
	fsr2.aCB = vec2<f16>(0.0, 0.0);
	fsr2.aW = vec2<f16>(0.0, 0.0);
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(0.0, 1.0) - pph.xx, vec2<f16>(-1.0, -1.0) - pph.yy,
		dir, len2, lob, clp, bczzR.xy, bczzG.xy, bczzB.xy); // b c
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(-1.0, 0.0) - pph.xx, vec2<f16>(1.0, 1.0) - pph.yy,
		dir, len2, lob, clp, ijfeR.xy, ijfeG.xy, ijfeB.xy); // i j
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(0.0, -1.0) - pph.xx, vec2<f16>(0.0, 0.0) - pph.yy,
		dir, len2, lob, clp, ijfeR.zw, ijfeG.zw, ijfeB.zw); // f e
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(1.0, 2.0) - pph.xx, vec2<f16>(1.0, 1.0) - pph.yy,
		dir, len2, lob, clp, klhgR.xy, klhgG.xy, klhgB.xy); // k l
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(2.0, 1.0) - pph.xx, vec2<f16>(0.0, 0.0) - pph.yy,
		dir, len2, lob, clp, klhgR.zw, klhgG.zw, klhgB.zw); // h g
	fsr2 = FsrEasuTapH(fsr2, vec2<f16>(1.0, 0.0) - pph.xx, vec2<f16>(2.0, 2.0) - pph.yy,
		dir, len2, lob, clp, zzonR.zw, zzonG.zw, zzonB.zw); // o n
  //------------------------------------------------------------------------------------------------------------------------------
	// Reduce packed pairs, normalize and dering.
	var aC : vec3<f16> = vec3<f16>(fsr2.aCR.x + fsr2.aCR.y, fsr2.aCG.x + fsr2.aCG.y, fsr2.aCB.x + fsr2.aCB.y);
	var aW : f16 = fsr2.aW.x + fsr2.aW.y;
	var min4 : vec3<f16> = -vec3<f16>(bothR.x, bothG.x, bothB.x);
	var max4 : vec3<f16> = vec3<f16>(bothR.y, bothG.y, bothB.y);
	var c : vec3<f16> = min(max4, max(min4, aC * (f16(1.0) / aW)));

	return vec4<f32>(vec3<f32>(c), 1.0);
}
