# AMD FSR 1.0 — wgsl/WebGPU port (fixed)

A working wgsl port of AMD's FidelityFX Super Resolution 1.0, forked from
[firdawolf/AMD-FSR1-wgpu-shader](https://github.com/firdawolf/AMD-FSR1-wgpu-shader)
because the upstream is abandoned and the RCAS pass does not work as-is.

## What was broken in the upstream

The EASU pass was largely fine. The RCAS pass had four issues:

1. **Sampler used where `textureLoad` was needed.** RCAS reads exact integer
   pixel neighbors; sampling with filtering is wrong here.
2. **Pixel coordinates passed to `textureSample`.** `textureSample` expects
   normalized UVs in `[0, 1]`, not absolute pixel indices, so the result was
   garbage at non-trivial resolutions.
3. **Denoising factor always applied.** The denoise flag was ignored; the
   denoise term was unconditionally added.
4. **`let` at module scope.** A module-scope constant was declared with `let`,
   which doesn't compile in current wgsl (`const` is required).

## What this fork changes

- RCAS rewritten to use `textureLoad` with integer pixel coords.
- `textureSample` calls converted to proper UV space where sampling is
  actually appropriate.
- Denoise flag now respected.
- Module-scope constant changed from `let` to `const`.
- Compute paths left untouched — there's no real benefit over the full-screen
  quad version unless workgroup-local storage is used to share sampled pixels
  and avoid global memory reads.

## Status

Used in production in a WebGPU renderer at reduced render resolution with
upscaling always on. No known issues at the time of this writing.

A pull request was opened against the upstream but the project appears
abandoned and the PR was never reviewed.

## Caveats / open questions

If you spot a problem with any of the above fixes, or know of a maintained
wgsl FSR port, please open an issue.
