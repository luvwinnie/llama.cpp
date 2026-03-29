// TurboQuant + RotorQuant constants for dk=64 KV cache compression
// Generated from Lloyd-Max solver (sigma = 1/sqrt(64)) and random Clifford rotors (seed=42)
//
// TurboQuant: ICLR 2026, arXiv 2504.19874
// RotorQuant: Clifford Cl(3,0) rotor-based rotation (replaces dense d×d matrix)

#pragma once

#ifndef QK_TURBO
#define QK_TURBO 64
#endif
#define TURBO_N_GROUPS_DK64 22  // ceil(64/3) + 1 for partial group handling

// Lloyd-Max optimal centroids for N(0, 1/sqrt(64)), sigma = 0.125

static const float TURBO_CENTROIDS_2BIT[4] = {
    -0.1888022011f, -0.0565975043f, +0.0565975043f, +0.1888022011f
};

static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.2689932131f, -0.1679886598f, -0.0945006602f, -0.0306367724f,
    +0.0306367724f, +0.0945006602f, +0.1679886598f, +0.2689932131f
};

static const float TURBO_CENTROIDS_4BIT[16] = {
    -0.3415736985f, -0.2586271557f, -0.2022558007f, -0.1570289020f,
    -0.1177925590f, -0.0820948913f, -0.0485060384f, -0.0160493791f,
    +0.0160493791f, +0.0485060384f, +0.0820948913f, +0.1177925590f,
    +0.1570289020f, +0.2022558007f, +0.2586271557f, +0.3415736985f
};

static const float TURBO_CENTROIDS_5BIT[32] = {
    -0.4083955150f, -0.3373013679f, -0.2906944103f, -0.2546061095f,
    -0.2244340531f, -0.1980518523f, -0.1742890614f, -0.1524259881f,
    -0.1319841406f, -0.1126258416f, -0.0941009084f, -0.0762159902f,
    -0.0588157384f, -0.0417705485f, -0.0249681088f, -0.0083071991f,
    +0.0083071991f, +0.0249681088f, +0.0417705485f, +0.0588157384f,
    +0.0762159902f, +0.0941009084f, +0.1126258416f, +0.1319841406f,
    +0.1524259881f, +0.1742890614f, +0.1980518523f, +0.2244340531f,
    +0.2546061095f, +0.2906944103f, +0.3373013679f, +0.4083955150f
};

// Clifford Cl(3,0) rotors for dk=64: 22 groups
// Each rotor: [scalar, bivector_12, bivector_13, bivector_23]
// R = s + b12*e12 + b13*e13 + b23*e23, normalized: s²+b12²+b13²+b23² = 1
// Sandwich product R*v*R̃ rotates 3D vector v
static const float TURBO_ROTORS_DK64[22][4] = {
    { -0.5810758132f, +0.1881336165f, -0.6420905950f, +0.4633317489f },
    { -0.7825166108f, -0.5171092835f, -0.3451342732f, +0.0338832737f },
    { -0.9736493913f, -0.0031270496f, -0.1587694595f, +0.1636745034f },
    { +0.7558195297f, +0.0353772814f, +0.6039422708f, +0.2504775837f },
    { -0.4019412537f, +0.2497972677f, -0.6495613409f, +0.5950753045f },
    { -0.9441569104f, -0.0431524588f, -0.1589495210f, +0.2853780715f },
    { +0.9905461835f, -0.0764433896f, -0.0628450472f, +0.0950008764f },
    { -0.9947952577f, +0.0189164406f, +0.0197454712f, +0.0981564056f },
    { +0.8280129269f, -0.2515164119f, -0.3995707213f, +0.3024521880f },
    { -0.5085383517f, -0.0829618363f, -0.6116935433f, -0.6002808406f },
    { -0.8177238211f, +0.3766284211f, +0.2752320033f, -0.3372330489f },
    { +0.9051578630f, +0.0547467075f, +0.1026044274f, +0.4088573988f },
    { -0.4950516186f, +0.7960640791f, +0.0792402342f, +0.3390086461f },
    { -0.2142821369f, -0.9099235733f, -0.1996192662f, -0.2937250502f },
    { +0.0906792127f, -0.1566348162f, +0.8510501620f, -0.4929061133f },
    { -0.1675309144f, -0.9625876778f, -0.1915514847f, +0.0930934156f },
    { +0.1975011320f, +0.6219078577f, +0.6937154677f, -0.3049307615f },
    { +0.7418692639f, +0.3713542950f, -0.0828014916f, -0.5521502476f },
    { -0.4870210726f, -0.7612370332f, +0.4116115383f, +0.1179177500f },
    { +0.2898732016f, -0.5283202099f, +0.1960426075f, +0.7735751928f },
    { +0.9602517260f, +0.1444977431f, -0.2093957743f, -0.1148496178f },
    { -0.0032822519f, -0.8704134794f, +0.3544517134f, -0.3416629693f },
};

// Rotor sandwich for 3D vector: v' = R * v * R_rev
// R = [s, b12, b13, b23], R_rev = [s, -b12, -b13, -b23]
// Expands to rotation matrix applied to (x, y, z)
static inline void turbo_rotor_forward(const float rotor[4], const float *v, float *out) {
    const float s = rotor[0], b12 = rotor[1], b13 = rotor[2], b23 = rotor[3];
    const float aa = s*s, bb = b12*b12, cc = b13*b13, dd = b23*b23;
    const float x = v[0], y = v[1], z = v[2];

    out[0] = (aa+bb-cc-dd)*x + 2.0f*(b12*b13 - s*b23)*y   + 2.0f*(b12*b23 + s*b13)*z;
    out[1] = 2.0f*(b12*b13 + s*b23)*x + (aa-bb+cc-dd)*y    + 2.0f*(b13*b23 - s*b12)*z;
    out[2] = 2.0f*(b12*b23 - s*b13)*x + 2.0f*(b13*b23 + s*b12)*y + (aa-bb-cc+dd)*z;
}

// Inverse rotor sandwich: v = R_rev * v' * R (transpose of rotation matrix)
static inline void turbo_rotor_inverse(const float rotor[4], const float *v, float *out) {
    const float s = rotor[0], b12 = rotor[1], b13 = rotor[2], b23 = rotor[3];
    const float aa = s*s, bb = b12*b12, cc = b13*b13, dd = b23*b23;
    const float x = v[0], y = v[1], z = v[2];

    // Transpose of forward rotation matrix
    out[0] = (aa+bb-cc-dd)*x + 2.0f*(b12*b13 + s*b23)*y   + 2.0f*(b12*b23 - s*b13)*z;
    out[1] = 2.0f*(b12*b13 - s*b23)*x + (aa-bb+cc-dd)*y    + 2.0f*(b13*b23 + s*b12)*z;
    out[2] = 2.0f*(b12*b23 + s*b13)*x + 2.0f*(b13*b23 - s*b12)*y + (aa-bb-cc+dd)*z;
}

// Find nearest centroid index (linear search, fine for ≤32 levels)
static inline int turbo_nearest_centroid(float val, const float *centroids, int n) {
    int best = 0;
    float best_dist = fabsf(val - centroids[0]);
    for (int i = 1; i < n; i++) {
        float dist = fabsf(val - centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}
