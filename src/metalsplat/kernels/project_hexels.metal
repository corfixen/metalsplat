// Per-primitive screen footprint of each Hexel's alpha >= 1/255 support,
// for tile binning. One thread per primitive. Mirrors
// metalsplat.reference.hexels_ref.hexel_rects / support_apothem exactly
// (port of hexel-kernel's compute_hex_bounds); that module is the oracle.
// Non-differentiable, like all of tiling.
#include <metal_stdlib>
using namespace metal;

constant float ALPHA_MIN = 1.0 / 255.0;
constant float LOG_254 = 5.537334267;
constant float HEX_RADIUS = 1.15470053838;  // 2 / sqrt(3)
constant float FLT_EPS = 1.1920928955078125e-07;
constant float LEGACY_TILE = 16.0;
constant float FILTER_SIZE = 0.707106;

// hexel_rects_ref.legacy_tile_window: hexel-kernel's historical tile window
// (compute_aabb + get_rect), as continuous bounds selecting exactly its
// 16 px tiles. CUDA pixel frame: rows shifted by -0.5 * row2.
inline bool legacy_window(float3 row0, float3 row1, float3 row2, float s_min, float op,
                          float width, float height, thread float2& lo, thread float2& hi) {
    float3 r0 = row0 - 0.5 * row2, r1 = row1 - 0.5 * row2;
    float cut = HEX_RADIUS * (1.0 + LOG_254 / s_min);
    float3 t = float3(cut * cut, cut * cut, -1.0);
    float d = dot(t, row2 * row2);
    bool bounded = d < 0.0;
    if (d == 0.0) return false;
    float3 f = t / d;
    float2 p = float2(dot(f, r0 * row2), dot(f, r1 * row2));
    float2 h0 = p * p - float2(dot(f, r0 * r0), dot(f, r1 * r1));
    float2 h = sqrt(max(float2(1e-4), h0));
    float radius = ceil(max(max(h.x, h.y), cut * FILTER_SIZE));
    float2 grid = float2(ceil(width / LEGACY_TILE), ceil(height / LEGACY_TILE));
    float2 rr = float2(radius);
    float2 a = min(max(trunc((p - rr) / LEGACY_TILE), 0.0), grid);
    float2 b = min(max(trunc((p + rr + LEGACY_TILE - 1.0) / LEGACY_TILE), 0.0), grid);
    if (!(a.x < b.x && a.y < b.y)) return false;
    if (bounded && op <= 1.0) {
        rr = min(rr, ceil(max(h, float2(cut * FILTER_SIZE))) + 1.0);
        a = min(max(trunc((p - rr) / LEGACY_TILE), 0.0), grid);
        b = min(max(trunc((p + rr + LEGACY_TILE - 1.0) / LEGACY_TILE), 0.0), grid);
    }
    lo = a * LEGACY_TILE + 0.5;
    hi = b * LEGACY_TILE - 0.5;
    return true;
}

kernel void hexel_rects(
    device const float* transform,   // (N,9) rows: x-num, y-num, z
    device const float* opacities,   // (N,)
    device const float* softness,    // (N,3)
    device const float* depths,      // (N,) center camera z
    constant float* fp,              // near, width, height, num_primitives, conservative
    device float* rects,             // (N,4) out; EMPTY_RECT when invalid
    device int* valid,               // (N,) out
    uint i [[thread_position_in_grid]])
{
    float near = fp[0], width = fp[1], height = fp[2];
    if (i >= uint(fp[3])) return;
    float op = opacities[i];
    float s0 = softness[i * 3 + 0], s1 = softness[i * 3 + 1], s2 = softness[i * 3 + 2];
    float s_min = min(s0, min(s1, s2)), s_max = max(s0, max(s1, s2));

    float margin;
    if (op > ALPHA_MIN) {
        float log_cutoff = log(max(255.0 * (1.0 + 8.0 * FLT_EPS) * op - 1.0, 1e-30));
        margin = log_cutoff / (log_cutoff < 0.0 ? s_max : s_min);
    } else {
        margin = LOG_254 / s_min;
    }
    float a = 1.0 + margin;
    a += 4e-5 * max(fabs(a), 1.0);
    float r = HEX_RADIUS * a;

    float3 row0 = float3(transform[i * 9 + 0], transform[i * 9 + 1], transform[i * 9 + 2]);
    float3 row1 = float3(transform[i * 9 + 3], transform[i * 9 + 4], transform[i * 9 + 5]);
    float3 row2 = float3(transform[i * 9 + 6], transform[i * 9 + 7], transform[i * 9 + 8]);
    float2 unit[6] = {float2(1.0, 0.0), float2(0.5, 1.0), float2(-0.5, 1.0),
                      float2(-1.0, 0.0), float2(-0.5, -1.0), float2(0.5, -1.0)};
    float3 hom[6];
    float z[6];
    for (int k = 0; k < 6; k++) {
        hom[k] = float3(unit[k].x * r, unit[k].y * a, 1.0);
        z[k] = dot(row2, hom[k]);
    }
    float lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY;
    for (int k = 0; k < 6; k++) {
        int n = (k + 1) % 6;
        if (z[k] >= near) {
            float px = dot(row0, hom[k]) / z[k], py = dot(row1, hom[k]) / z[k];
            lo_x = min(lo_x, px); hi_x = max(hi_x, px);
            lo_y = min(lo_y, py); hi_y = max(hi_y, py);
        }
        if ((z[k] < near) != (z[n] < near)) {
            float t = (near - z[k]) / (z[n] - z[k]);
            float3 e = hom[k] + t * (hom[n] - hom[k]);
            float px = dot(row0, e) / near, py = dot(row1, e) / near;
            lo_x = min(lo_x, px); hi_x = max(hi_x, px);
            lo_y = min(lo_y, py); hi_y = max(hi_y, py);
        }
    }
    bool has = a > 0.0 && isfinite(lo_x) && isfinite(hi_x)
        && lo_x <= width && lo_y <= height && hi_x >= 0.0 && hi_y >= 0.0;
    lo_x -= 1.0; lo_y -= 1.0; hi_x += 1.0; hi_y += 1.0;
    if (has && fp[4] == 0.0) {
        float2 wlo, whi;
        has = legacy_window(row0, row1, row2, s_min, op, width, height, wlo, whi);
        lo_x = max(lo_x, wlo.x); lo_y = max(lo_y, wlo.y);
        hi_x = min(hi_x, whi.x); hi_y = min(hi_y, whi.y);
        has = has && hi_x >= lo_x && hi_y >= lo_y;
    }
    bool ok = has && depths[i] > near && op >= ALPHA_MIN;
    valid[i] = ok ? 1 : 0;
    if (ok) {
        rects[i * 4 + 0] = lo_x; rects[i * 4 + 1] = lo_y;
        rects[i * 4 + 2] = hi_x; rects[i * 4 + 3] = hi_y;
    } else {
        rects[i * 4 + 0] = 0.0; rects[i * 4 + 1] = 0.0;
        rects[i * 4 + 2] = -1.0; rects[i * 4 + 3] = -1.0;
    }
}
