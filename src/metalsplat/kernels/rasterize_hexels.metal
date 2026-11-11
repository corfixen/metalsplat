// Tile-based front-to-back rasterizer for 3D Hexels (hexagonal surfels),
// forward + backward. The math mirrors
// metalsplat.reference.hexels_ref.rasterize_hexels exactly; that module is
// the oracle these kernels are tested against, and its docstring lists the
// CUDA sources (honeycomb-surfels/hexel-kernel) the formulas come from.
//
// Dispatch: one threadgroup per tile_size x tile_size tile, one thread per
// pixel, as in rasterize_2dgs.metal. Every lane of a SIMD group belongs to
// the same tile, so all lanes walk the same primitive list in lockstep:
// lanes whose pixel is finished (or off-image) keep iterating with zero
// contributions, which lets per-primitive quantities be reduced with
// simd_sum and committed with one atomic per SIMD group instead of one per
// pixel. Loops exit once simd_all(done).
//
// Per-pixel ray/primitive intersection: the per-primitive `transform` is
// project_2dgs_ref's 3x3 (rows: x-numerator, y-numerator, z; columns: t_u,
// t_v, mean). For a pixel (px, py) the two pulled-back planes
// hu = row0 - px*row2 and hv = row1 - py*row2 meet in the local point
// cross(hu, hv) / cross(hu, hv).z = (u, v, 1), where (u, v) are
// scale-normalized coordinates in which the hexagon has unit apothem.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant float FAR_PLANE = 100.0;
constant float ALPHA_MAX = 0.99;
constant float ALPHA_MIN = 1.0 / 255.0;
constant float T_MIN = 1e-4;
constant float HEX_A = 0.8660254;  // sqrt(3) / 2
constant float HEX_B = 0.5;

// Packed per-pixel float outputs of the forward pass (channel-last).
#define F_IMAGE 0       // 3
#define F_ALPHA 3
#define F_DEPTH 4
#define F_NORMAL 5      // 3
#define F_DIST 8
#define F_MEDIAN 9
#define F_OVERDRAW 10
#define F_HITS 11
#define F_M1 12         // backward-only: sum of w*m
#define F_M2 13         // backward-only: sum of w*m*m
#define F_T 14          // final transmittance
#define F_COUNT 15

// Packed per-pixel int outputs: list indices, -1 for none.
#define I_LAST_ACC 0    // last accumulated primitive
#define I_LAST_HIT 1    // last visited primitive (accumulated or terminating)
#define I_MEDIAN 2      // primitive that set the median depth
#define I_COUNT 3

// Packed incoming per-pixel gradients for the backward pass.
#define G_IMAGE 0       // 3
#define G_ALPHA 3
#define G_DEPTH 4
#define G_NORMAL 5      // 3
#define G_DIST 8
#define G_MEDIAN 9
#define G_OVERDRAW 10
#define G_COUNT 11

// iparams: tiles_x, width, height, tile_size, color_model, return_contrib,
//          return_owner, num_primitives
// fparams: near, background r, g, b

struct HexHit {
    float3 hu;
    float3 hv;
    float wloc;
    float u;
    float v;
    float z;
    float dist;
    int edge;
    float softness;
    float boundary;
    float raw;
    float alpha;
    bool hit;
};

// hex_sdf_edge: max of the three oriented half-plane distances, with the
// CUDA tie convention (second slant wins a slant tie; a slant wins a tie
// against the horizontal edge).
inline float hex_sdf(float u, float v, thread int& edge) {
    float d0 = fabs(v);
    float d1 = fabs(HEX_A * u + HEX_B * v);
    float d2 = fabs(HEX_A * u - HEX_B * v);
    float slant = d1;
    edge = 1;
    if (d2 >= slant) { slant = d2; edge = 2; }
    if (d0 > slant) { edge = 0; return d0 - 1.0; }
    return slant - 1.0;
}

// d(sdf)/d(u, v) for the edge family `edge` selected by hex_sdf.
inline float2 hex_sdf_grad(float u, float v, int edge) {
    if (edge == 0) return float2(0.0, v >= 0.0 ? 1.0 : -1.0);
    if (edge == 1) {
        float s = (HEX_A * u + HEX_B * v) >= 0.0 ? 1.0 : -1.0;
        return float2(HEX_A * s, HEX_B * s);
    }
    float s = (HEX_A * u - HEX_B * v) >= 0.0 ? 1.0 : -1.0;
    return float2(HEX_A * s, -HEX_B * s);
}

inline float stable_sigmoid(float x) {
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    float e = exp(x);
    return e / (1.0 + e);
}

inline HexHit eval_hexel(
    device const float* transform, device const float* opacities,
    device const float* softness, int gid, float px, float py, float near)
{
    HexHit h;
    float3 row0 = float3(transform[gid * 9 + 0], transform[gid * 9 + 1], transform[gid * 9 + 2]);
    float3 row1 = float3(transform[gid * 9 + 3], transform[gid * 9 + 4], transform[gid * 9 + 5]);
    float3 row2 = float3(transform[gid * 9 + 6], transform[gid * 9 + 7], transform[gid * 9 + 8]);
    h.hu = row0 - px * row2;
    h.hv = row1 - py * row2;
    float3 cr = cross(h.hu, h.hv);
    h.wloc = cr.z;
    h.hit = false;
    h.u = 0.0; h.v = 0.0; h.z = 0.0; h.dist = 0.0; h.edge = 0;
    h.softness = 1.0; h.boundary = 0.0; h.raw = 0.0; h.alpha = 0.0;
    if (h.wloc == 0.0) return h;
    h.u = cr.x / h.wloc;
    h.v = cr.y / h.wloc;
    h.z = row2.x * h.u + row2.y * h.v + row2.z;
    if (!(h.z >= near)) return h;
    h.dist = hex_sdf(h.u, h.v, h.edge);
    h.softness = softness[gid * 3 + h.edge];
    h.boundary = stable_sigmoid(-h.dist * h.softness);
    h.raw = opacities[gid] * h.boundary;
    h.alpha = min(h.raw, ALPHA_MAX);
    h.hit = h.alpha >= ALPHA_MIN;
    return h;
}

// quadratic_bounded's coordinates: radially pulled onto the hexagon
// outside it.
inline float2 bounded_coords(float u, float v) {
    int e;
    float rho = hex_sdf(u, v, e) + 1.0;
    float inv = rho > 1.0 ? 1.0 / rho : 1.0;
    return float2(u * inv, v * inv);
}

// VJP of bounded_coords: gradient on (u, v) from gradient on the bounded
// coordinates.
inline float2 bounded_coords_vjp(float u, float v, float2 g) {
    int e;
    float rho = hex_sdf(u, v, e) + 1.0;
    if (rho <= 1.0) return g;
    float2 drho = hex_sdf_grad(u, v, e);
    float inv = 1.0 / rho;
    float radial = (g.x * u + g.y * v) * inv * inv;
    return float2(g.x * inv - radial * drho.x, g.y * inv - radial * drho.y);
}

inline float3 load3(device const float* p, int i, int k) {
    return float3(p[i * k + 0], p[i * k + 1], p[i * k + 2]);
}

inline float3 eval_color(
    int color_model, device const float* colors, device const float* cgrads,
    device const float* cquads, int gid, float u, float v)
{
    float3 c = load3(colors, gid, 3);
    if (color_model >= 1) {
        c += u * float3(cgrads[gid * 6 + 0], cgrads[gid * 6 + 1], cgrads[gid * 6 + 2])
           + v * float3(cgrads[gid * 6 + 3], cgrads[gid * 6 + 4], cgrads[gid * 6 + 5]);
    }
    if (color_model >= 2) {
        float2 q = color_model == 2 ? float2(u, v) : bounded_coords(u, v);
        c += (q.x * q.x) * float3(cquads[gid * 9 + 0], cquads[gid * 9 + 1], cquads[gid * 9 + 2])
           + (q.x * q.y) * float3(cquads[gid * 9 + 3], cquads[gid * 9 + 4], cquads[gid * 9 + 5])
           + (q.y * q.y) * float3(cquads[gid * 9 + 6], cquads[gid * 9 + 7], cquads[gid * 9 + 8]);
    }
    return c;
}

inline float normalized_depth(float z, float near) {
    return FAR_PLANE / (FAR_PLANE - near) * (1.0 - near / z);
}

inline float d_normalized_depth_dz(float z, float near) {
    return FAR_PLANE / (FAR_PLANE - near) * near / (z * z);
}

inline void commit(device atomic_float* dst, int index, float value, uint lane) {
    float total = simd_sum(value);
    if (lane == 0 && total != 0.0) {
        atomic_fetch_add_explicit(&dst[index], total, memory_order_relaxed);
    }
}

kernel void rasterize_hexels_forward(
    device const float* transform,      // (N,9)
    device const float* opacities,      // (N,)
    device const float* softness,       // (N,3)
    device const float* colors,         // (N,3)
    device const float* cgrads,         // (N,6)
    device const float* cquads,         // (N,9)
    device const float* normals,        // (N,3) camera-facing
    device const int* sorted_ids,       // (M,)
    device const int* tile_bins,        // (num_tiles,2)
    constant int* ip,
    constant float* fp,
    device float* out_f,                // (H,W,F_COUNT)
    device int* out_i,                  // (H,W,I_COUNT)
    device atomic_float* contrib,       // (N,) when return_contrib
    device float* owner_f,              // (H,W,5) weight, u, v, sdf, depth when return_owner
    device int* owner_i,                // (H,W) when return_owner
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int tiles_x = ip[0], width = ip[1], height = ip[2], tile_size = ip[3];
    int color_model = ip[4];
    bool return_contrib = ip[5] != 0, return_owner = ip[6] != 0;
    float near = fp[0];
    float3 background = float3(fp[1], fp[2], fp[3]);

    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    bool inside = px < uint(width) && py < uint(height);
    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    int end = tile_bins[tile_id * 2 + 1];
    float pcx = float(px) + 0.5, pcy = float(py) + 0.5;

    float3 C = float3(0.0);
    float T = 1.0;
    float D = 0.0, A = 0.0, M1 = 0.0, M2 = 0.0, L = 0.0, median = 0.0;
    float3 N = float3(0.0);
    float overdraw = 0.0, hits = 0.0;
    int last_acc = -1, last_hit = -1, median_idx = -1;
    int owner = -1;
    float owner_w = 0.0, owner_u = 0.0, owner_v = 0.0, owner_sdf = 0.0, owner_z = 0.0;
    bool done = !inside;

    for (int idx = start; idx < end; idx++) {
        if (simd_all(done)) break;
        int gid = sorted_ids[idx];
        float w = 0.0;
        if (!done) {
            HexHit h = eval_hexel(transform, opacities, softness, gid, pcx, pcy, near);
            if (h.hit) {
                overdraw += h.raw;
                hits += 1.0;
                last_hit = idx;
                float test_T = T * (1.0 - h.alpha);
                if (test_T < T_MIN) {
                    done = true;
                } else {
                    w = h.alpha * T;
                    float3 c = eval_color(color_model, colors, cgrads, cquads, gid, h.u, h.v);
                    C += w * c;
                    if (T > 0.5) { median = h.z; median_idx = idx; }
                    float m = normalized_depth(h.z, near);
                    L += w * (m * m * A + M2 - 2.0 * m * M1);
                    A += w;
                    M1 += w * m;
                    M2 += w * m * m;
                    D += w * h.z;
                    N += w * load3(normals, gid, 3);
                    if (return_owner && w > owner_w) {
                        owner = gid; owner_w = w; owner_u = h.u; owner_v = h.v;
                        owner_sdf = h.dist; owner_z = h.z;
                    }
                    T = test_T;
                    last_acc = idx;
                }
            }
        }
        if (return_contrib) commit(contrib, gid, w, lane);
    }

    if (!inside) return;
    uint pix = py * uint(width) + px;
    device float* o = out_f + pix * F_COUNT;
    float3 img = C + T * background;
    o[F_IMAGE + 0] = img.x; o[F_IMAGE + 1] = img.y; o[F_IMAGE + 2] = img.z;
    o[F_ALPHA] = 1.0 - T;
    o[F_DEPTH] = D;
    o[F_NORMAL + 0] = N.x; o[F_NORMAL + 1] = N.y; o[F_NORMAL + 2] = N.z;
    o[F_DIST] = L;
    o[F_MEDIAN] = median;
    o[F_OVERDRAW] = overdraw;
    o[F_HITS] = hits;
    o[F_M1] = M1;
    o[F_M2] = M2;
    o[F_T] = T;
    device int* oi = out_i + pix * I_COUNT;
    oi[I_LAST_ACC] = last_acc;
    oi[I_LAST_HIT] = last_hit;
    oi[I_MEDIAN] = median_idx;
    if (return_owner) {
        owner_i[pix] = owner;
        owner_f[pix * 5 + 0] = owner_w;
        owner_f[pix * 5 + 1] = owner_u;
        owner_f[pix * 5 + 2] = owner_v;
        owner_f[pix * 5 + 3] = owner_sdf;
        owner_f[pix * 5 + 4] = owner_z;
    }
}

// Backward. Walks each pixel's list back-to-front from its last visited
// primitive, recovering the transmittance in front of every accumulated
// primitive as T_before = T_after / (1 - alpha) (alpha <= 0.99, never
// singular), exactly as rasterize_2dgs.metal does.
//
// Per accumulated primitive m (weight w = alpha*T_before), the gradient on
// alpha collects:
//   image:      T_before * (c - A_color) . dC   (A_color = composited color
//               behind m, seeded with the background)
//   depth:      T_before * (z - A_depth) * dD
//   normal:     T_before * (n - A_normal) . dN
//   alpha out:  T_final / (1 - alpha) * dAlpha  (alpha_out = 1 - T_final)
//   distortion: rasterize_2dgs.metal's closed form (same accumulation)
// Unclamped alpha (raw <= 0.99) passes that gradient on to opacity and the
// boundary sigmoid; the overdraw map (a sum of *unclamped* alpha over every
// visited primitive, including the one that triggers termination) adds its
// own gradient regardless of clamping, as hexel-kernel's backward does.
//
// (u, v) gradients -- from the color field, the boundary SDF, and the
// intersection depth z = row2 . (u, v, 1) -- are pulled back onto the
// transform rows with the implicit-function adjoint on the two ray-plane
// equations (see rasterize_2dgs.metal for the derivation).
//
// Gradient commits: the tile walks its list in batches of STAGE_BATCH
// entries. Per entry, each SIMD group reduces its lanes with simd_sum into
// its own rows of a threadgroup stage; after the batch, all threads flush
// the stage with one global atomic per (primitive, gradient) per tile.
// Committing per SIMD group straight to device memory instead (8 serial
// atomics per primitive per tile from one lane) made the backward ~1.5x
// slower on a 31k-primitive DTU scene.

// Backward gradient staging: per batch of STAGE_BATCH list entries, slot
// layout of the per-primitive gradients in threadgroup memory.
#define STAGE_BATCH 16
#define MAX_SG 8    // SIMD groups per tile: tile_size <= 16
#define S_T 0       // transform, 9
#define S_OP 9
#define S_SOFT 10   // 3
#define S_C 13      // 3
#define S_GU 16     // 6
#define S_Q 22      // 9
#define S_N 31      // 3
#define S_ABS 34
#define S_PIX 35
#define S_COUNT 36

// Each SIMD group owns its own stage rows, so lane 0 accumulates with a
// plain add (Metal has no threadgroup float atomics).
inline void stage_add(threadgroup float* st, int slot, float value, uint lane) {
    float total = simd_sum(value);
    if (lane == 0) st[slot] += total;
}

kernel void rasterize_hexels_backward(
    device const float* transform,
    device const float* opacities,
    device const float* softness,
    device const float* colors,
    device const float* cgrads,
    device const float* cquads,
    device const float* normals,
    device const float* depths,         // (N,) center depth, scales the screen-space signal
    device const int* sorted_ids,
    device const int* tile_bins,
    constant int* ip,
    constant float* fp,
    device const float* out_f,          // forward's packed per-pixel floats
    device const int* out_i,            // forward's packed per-pixel ints
    device const float* grad_f,         // (H,W,G_COUNT)
    device atomic_float* d_transform,   // (N,9)
    device atomic_float* d_opacities,   // (N,)
    device atomic_float* d_softness,    // (N,3)
    device atomic_float* d_colors,      // (N,3)
    device atomic_float* d_cgrads,      // (N,6)
    device atomic_float* d_cquads,      // (N,9)
    device atomic_float* d_normals,     // (N,3)
    device atomic_float* acc_abs,       // (N,) sum over pixels of |d L / d means2d|
    device atomic_float* acc_pixels,    // (N,) pixels contributing to acc_abs
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]],
    uint local_idx [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    int tiles_x = ip[0], width = ip[1], height = ip[2], tile_size = ip[3];
    int color_model = ip[4];
    bool want_normals = ip[5] != 0, want_accum = ip[6] != 0;
    uint nthreads = uint(tile_size * tile_size);
    float near = fp[0];
    float3 background = float3(fp[1], fp[2], fp[3]);

    uint px = tg_pos.x * uint(tile_size) + local_pos.x;
    uint py = tg_pos.y * uint(tile_size) + local_pos.y;
    bool inside = px < uint(width) && py < uint(height);
    uint tile_id = tg_pos.y * uint(tiles_x) + tg_pos.x;
    int start = tile_bins[tile_id * 2 + 0];
    uint pix = inside ? py * uint(width) + px : 0;
    float pcx = float(px) + 0.5, pcy = float(py) + 0.5;

    int last_acc = inside ? out_i[pix * I_COUNT + I_LAST_ACC] : -1;
    int last_hit = inside ? out_i[pix * I_COUNT + I_LAST_HIT] : -1;
    int median_idx = inside ? out_i[pix * I_COUNT + I_MEDIAN] : -1;

    device const float* o = out_f + pix * F_COUNT;
    device const float* g = grad_f + pix * G_COUNT;
    float3 dC = inside ? float3(g[G_IMAGE + 0], g[G_IMAGE + 1], g[G_IMAGE + 2]) : float3(0.0);
    float dAlpha = inside ? g[G_ALPHA] : 0.0;
    float dD = inside ? g[G_DEPTH] : 0.0;
    float3 dN = inside ? float3(g[G_NORMAL + 0], g[G_NORMAL + 1], g[G_NORMAL + 2]) : float3(0.0);
    float dDist = inside ? g[G_DIST] : 0.0;
    float dMedian = inside ? g[G_MEDIAN] : 0.0;
    float dOverdraw = inside ? g[G_OVERDRAW] : 0.0;

    float T_final = inside ? o[F_T] : 1.0;
    float T = T_final;
    float3 A_color = background;
    float A_depth = 0.0;
    float3 A_normal = float3(0.0);
    float A_run = inside ? (1.0 - T_final) : 0.0;
    float M1_run = inside ? o[F_M1] : 0.0;
    float M2_run = inside ? o[F_M2] : 0.0;
    float Asuf = 0.0, M1suf = 0.0, M2suf = 0.0, Lsuf = 0.0;

    // Tile-wide upper bound of the list walk, so every thread runs the
    // same batches and can meet at the batch barriers.
    threadgroup int tg_hi[32];
    threadgroup float stage[MAX_SG * STAGE_BATCH * S_COUNT];
    uint nsg = nthreads / 32;
    int sg_hi = simd_max(last_hit);
    if (lane == 0) tg_hi[sg] = sg_hi;
    for (uint k = local_idx; k < MAX_SG * STAGE_BATCH * S_COUNT; k += nthreads)
        stage[k] = 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int hi = -1;
    for (uint k = 0; k < nsg; k++) hi = max(hi, tg_hi[k]);

    for (int batch_end = hi + 1; batch_end > start; batch_end -= STAGE_BATCH) {
      int batch_start = max(start, batch_end - STAGE_BATCH);
      for (int idx = batch_end - 1; idx >= batch_start; idx--) {
        int gid = sorted_ids[idx];
        float g_op = 0.0;
        float3 g_soft = float3(0.0);
        float3 g_row0 = float3(0.0), g_row1 = float3(0.0), g_row2 = float3(0.0);
        float3 g_c = float3(0.0), g_gu = float3(0.0), g_gv = float3(0.0);
        float3 g_quu = float3(0.0), g_quv = float3(0.0), g_qvv = float3(0.0);
        float3 g_n = float3(0.0);
        float g_abs = 0.0, g_pix = 0.0;
        bool touched = false;

        if (idx <= last_hit) {
            HexHit h = eval_hexel(transform, opacities, softness, gid, pcx, pcy, near);
            if (h.hit) {
                touched = true;
                bool accumulated = idx <= last_acc;
                float d_u = 0.0, d_v = 0.0, g_z = 0.0;
                float d_alpha_comp = 0.0;
                if (accumulated) {
                    float inv_1ma = 1.0 / (1.0 - h.alpha);
                    float T_before = T * inv_1ma;
                    float w = h.alpha * T_before;
                    float3 c = eval_color(color_model, colors, cgrads, cquads, gid, h.u, h.v);
                    float3 n = load3(normals, gid, 3);

                    d_alpha_comp = dot(T_before * (c - A_color), dC)
                                 + T_before * (h.z - A_depth) * dD
                                 + dot(T_before * (n - A_normal), dN)
                                 + T_final * inv_1ma * dAlpha;

                    float m = normalized_depth(h.z, near);
                    float m_sq = m * m;
                    float A_excl = A_run - w;
                    float M1_excl = M1_run - w * m;
                    float M2_excl = M2_run - w * m_sq;
                    float Fpre = m_sq * A_excl - 2.0 * m * M1_excl + M2_excl;
                    float Gsuf = m_sq * Asuf - 2.0 * m * M1suf + M2suf;
                    float crs = M2_excl * Asuf - 2.0 * M1_excl * M1suf + A_excl * M2suf;
                    d_alpha_comp += (T_before * Fpre
                                   + T_before * Gsuf * (1.0 - 2.0 * h.alpha) * inv_1ma
                                   - 2.0 * Lsuf * inv_1ma
                                   - crs * inv_1ma) * dDist;
                    g_z = w * dD
                        + 2.0 * w * (m * (A_excl + Asuf) - (M1_excl + M1suf))
                              * dDist * d_normalized_depth_dz(h.z, near);
                    if (idx == median_idx) g_z += dMedian;

                    float3 wdC = w * dC;
                    g_c = wdC;
                    g_n = w * dN;
                    if (color_model >= 1) {
                        float3 gu = float3(cgrads[gid * 6 + 0], cgrads[gid * 6 + 1], cgrads[gid * 6 + 2]);
                        float3 gv = float3(cgrads[gid * 6 + 3], cgrads[gid * 6 + 4], cgrads[gid * 6 + 5]);
                        g_gu = h.u * wdC;
                        g_gv = h.v * wdC;
                        d_u += dot(wdC, gu);
                        d_v += dot(wdC, gv);
                    }
                    if (color_model >= 2) {
                        float3 quu = float3(cquads[gid * 9 + 0], cquads[gid * 9 + 1], cquads[gid * 9 + 2]);
                        float3 quv = float3(cquads[gid * 9 + 3], cquads[gid * 9 + 4], cquads[gid * 9 + 5]);
                        float3 qvv = float3(cquads[gid * 9 + 6], cquads[gid * 9 + 7], cquads[gid * 9 + 8]);
                        float2 q = color_model == 2 ? float2(h.u, h.v) : bounded_coords(h.u, h.v);
                        g_quu = (q.x * q.x) * wdC;
                        g_quv = (q.x * q.y) * wdC;
                        g_qvv = (q.y * q.y) * wdC;
                        float2 dq = float2(dot(wdC, 2.0 * q.x * quu + q.y * quv),
                                           dot(wdC, q.x * quv + 2.0 * q.y * qvv));
                        if (color_model == 3) dq = bounded_coords_vjp(h.u, h.v, dq);
                        d_u += dq.x;
                        d_v += dq.y;
                    }

                    A_color = h.alpha * c + (1.0 - h.alpha) * A_color;
                    A_depth = h.alpha * h.z + (1.0 - h.alpha) * A_depth;
                    A_normal = h.alpha * n + (1.0 - h.alpha) * A_normal;
                    Lsuf += w * Gsuf;
                    Asuf += w;
                    M1suf += w * m;
                    M2suf += w * m_sq;
                    A_run = A_excl;
                    M1_run = M1_excl;
                    M2_run = M2_excl;
                    T = T_before;
                }

                float d_raw = (h.raw > ALPHA_MAX ? 0.0 : d_alpha_comp) + dOverdraw;
                float bb = h.boundary * (1.0 - h.boundary);
                float op = opacities[gid];
                g_op = d_raw * h.boundary;
                float g_s = d_raw * op * (-h.dist) * bb;
                if (h.edge == 0) g_soft.x = g_s; else if (h.edge == 1) g_soft.y = g_s; else g_soft.z = g_s;
                float d_dist = d_raw * op * (-h.softness) * bb;
                float2 dsdf = hex_sdf_grad(h.u, h.v, h.edge);
                d_u += d_dist * dsdf.x;
                d_v += d_dist * dsdf.y;

                float3 row2 = float3(transform[gid * 9 + 6], transform[gid * 9 + 7], transform[gid * 9 + 8]);
                d_u += g_z * row2.x;
                d_v += g_z * row2.y;
                float3 uv1 = float3(h.u, h.v, 1.0);
                float lam1 = (d_u * h.hv.y - h.hv.x * d_v) / h.wloc;
                float lam2 = (h.hu.x * d_v - h.hu.y * d_u) / h.wloc;
                g_row0 = -lam1 * uv1;
                g_row1 = -lam2 * uv1;
                g_row2 = (lam1 * pcx + lam2 * pcy) * uv1 + g_z * uv1;

                // Screen-space signal for densification, in pixels: row0[2]
                // = means2d.x * z_mean exactly, so d(means2d) at fixed depth
                // is the mean-column gradient times the center depth (the
                // same identity rasterize_2dgs.metal uses).
                g_abs = length(float2(g_row0.z, g_row1.z) * depths[gid]);
                g_pix = 1.0;
            }
        }

        // Uniform across the SIMD group: skip the reductions entirely when
        // no lane composited this primitive. Each SIMD group's sums go to the
        // on-chip stage; the batch flush below commits them globally.
        if (!simd_any(touched)) continue;
        threadgroup float* st = stage + (sg * STAGE_BATCH + uint(idx - batch_start)) * S_COUNT;
        stage_add(st, S_OP, g_op, lane);
        stage_add(st, S_SOFT + 0, g_soft.x, lane);
        stage_add(st, S_SOFT + 1, g_soft.y, lane);
        stage_add(st, S_SOFT + 2, g_soft.z, lane);
        stage_add(st, S_T + 0, g_row0.x, lane);
        stage_add(st, S_T + 1, g_row0.y, lane);
        stage_add(st, S_T + 2, g_row0.z, lane);
        stage_add(st, S_T + 3, g_row1.x, lane);
        stage_add(st, S_T + 4, g_row1.y, lane);
        stage_add(st, S_T + 5, g_row1.z, lane);
        stage_add(st, S_T + 6, g_row2.x, lane);
        stage_add(st, S_T + 7, g_row2.y, lane);
        stage_add(st, S_T + 8, g_row2.z, lane);
        stage_add(st, S_C + 0, g_c.x, lane);
        stage_add(st, S_C + 1, g_c.y, lane);
        stage_add(st, S_C + 2, g_c.z, lane);
        if (want_normals) {
            stage_add(st, S_N + 0, g_n.x, lane);
            stage_add(st, S_N + 1, g_n.y, lane);
            stage_add(st, S_N + 2, g_n.z, lane);
        }
        if (color_model >= 1) {
            stage_add(st, S_GU + 0, g_gu.x, lane);
            stage_add(st, S_GU + 1, g_gu.y, lane);
            stage_add(st, S_GU + 2, g_gu.z, lane);
            stage_add(st, S_GU + 3, g_gv.x, lane);
            stage_add(st, S_GU + 4, g_gv.y, lane);
            stage_add(st, S_GU + 5, g_gv.z, lane);
        }
        if (color_model >= 2) {
            stage_add(st, S_Q + 0, g_quu.x, lane);
            stage_add(st, S_Q + 1, g_quu.y, lane);
            stage_add(st, S_Q + 2, g_quu.z, lane);
            stage_add(st, S_Q + 3, g_quv.x, lane);
            stage_add(st, S_Q + 4, g_quv.y, lane);
            stage_add(st, S_Q + 5, g_quv.z, lane);
            stage_add(st, S_Q + 6, g_qvv.x, lane);
            stage_add(st, S_Q + 7, g_qvv.y, lane);
            stage_add(st, S_Q + 8, g_qvv.z, lane);
        }
        if (want_accum) {
            stage_add(st, S_ABS, g_abs, lane);
            stage_add(st, S_PIX, g_pix, lane);
        }
      }

      // Flush: one global atomic per (primitive, gradient) per tile, spread
      // over every thread, and reset the stage for the next batch.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint count = uint(batch_end - batch_start) * S_COUNT;
      for (uint k = local_idx; k < count; k += nthreads) {
          float v = 0.0;
          for (uint j = 0; j < nsg; j++) {
              v += stage[j * STAGE_BATCH * S_COUNT + k];
              stage[j * STAGE_BATCH * S_COUNT + k] = 0.0;
          }
          if (v == 0.0) continue;
          int gid = sorted_ids[batch_start + int(k / S_COUNT)];
          int s = int(k % S_COUNT);
          device atomic_float* dst;
          if (s < S_OP) dst = d_transform + gid * 9 + (s - S_T);
          else if (s == S_OP) dst = d_opacities + gid;
          else if (s < S_C) dst = d_softness + gid * 3 + (s - S_SOFT);
          else if (s < S_GU) dst = d_colors + gid * 3 + (s - S_C);
          else if (s < S_Q) dst = d_cgrads + gid * 6 + (s - S_GU);
          else if (s < S_N) dst = d_cquads + gid * 9 + (s - S_Q);
          else if (s < S_ABS) dst = d_normals + gid * 3 + (s - S_N);
          else if (s == S_ABS) dst = acc_abs + gid;
          else dst = acc_pixels + gid;
          atomic_fetch_add_explicit(dst, v, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
