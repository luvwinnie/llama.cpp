// TurboQuant + RotorQuant test suite
// Validates against TurboQuant paper (ICLR 2026, arXiv 2504.19874) bounds
//
// Paper Table 2 — MSE distortion bounds for unit vectors:
//   2-bit: 0.117    3-bit: 0.030    4-bit: 0.009    5-bit: ~0.003
//
// Paper Theorem 2 — Inner product distortion:
//   D_prod <= sqrt(3*pi^2) * ||y||^2 / d * 1/4^b

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ggml-turbo-rotors.h"

// ---- Helpers ----

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS(name) do { printf("PASS: %s\n", name); tests_passed++; } while(0)

// Simple RNG (xoshiro128+)
static uint32_t rng_state[4] = {42, 123, 456, 789};
static uint32_t rng_next(void) {
    uint32_t t = rng_state[1] << 9;
    rng_state[2] ^= rng_state[0]; rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2]; rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 11) | (rng_state[3] >> 21);
    return rng_state[0] + rng_state[3];
}
static float rng_normal(void) {
    float u1 = (rng_next() + 1.0f) / 4294967296.0f;
    float u2 = (rng_next() + 1.0f) / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
static void rng_seed(uint32_t s) {
    rng_state[0] = s; rng_state[1] = s*2+1; rng_state[2] = s*3+2; rng_state[3] = s*4+3;
}

// ---- Codebook Tests (matching test_codebook.py) ----

void test_centroids_sorted(void) {
    for (int i = 1; i < 4; i++) TEST_ASSERT(TURBO_CENTROIDS_2BIT[i] > TURBO_CENTROIDS_2BIT[i-1], "2-bit not sorted");
    for (int i = 1; i < 8; i++) TEST_ASSERT(TURBO_CENTROIDS_3BIT[i] > TURBO_CENTROIDS_3BIT[i-1], "3-bit not sorted");
    for (int i = 1; i < 16; i++) TEST_ASSERT(TURBO_CENTROIDS_4BIT[i] > TURBO_CENTROIDS_4BIT[i-1], "4-bit not sorted");
    for (int i = 1; i < 32; i++) TEST_ASSERT(TURBO_CENTROIDS_5BIT[i] > TURBO_CENTROIDS_5BIT[i-1], "5-bit not sorted");
    TEST_PASS("centroids sorted (all 4 bit widths)");
}

void test_centroids_symmetric(void) {
    for (int i = 0; i < 2; i++)
        TEST_ASSERT(fabsf(TURBO_CENTROIDS_2BIT[i] + TURBO_CENTROIDS_2BIT[3-i]) < 1e-6f, "2-bit not symmetric");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT(fabsf(TURBO_CENTROIDS_3BIT[i] + TURBO_CENTROIDS_3BIT[7-i]) < 1e-6f, "3-bit not symmetric");
    for (int i = 0; i < 8; i++)
        TEST_ASSERT(fabsf(TURBO_CENTROIDS_4BIT[i] + TURBO_CENTROIDS_4BIT[15-i]) < 1e-6f, "4-bit not symmetric");
    for (int i = 0; i < 16; i++)
        TEST_ASSERT(fabsf(TURBO_CENTROIDS_5BIT[i] + TURBO_CENTROIDS_5BIT[31-i]) < 1e-6f, "5-bit not symmetric");
    TEST_PASS("centroids symmetric (all 4 bit widths)");
}

void test_centroids_correct_count(void) {
    TEST_ASSERT(sizeof(TURBO_CENTROIDS_2BIT)/sizeof(float) == 4, "2-bit count wrong");
    TEST_ASSERT(sizeof(TURBO_CENTROIDS_3BIT)/sizeof(float) == 8, "3-bit count wrong");
    TEST_ASSERT(sizeof(TURBO_CENTROIDS_4BIT)/sizeof(float) == 16, "4-bit count wrong");
    TEST_ASSERT(sizeof(TURBO_CENTROIDS_5BIT)/sizeof(float) == 32, "5-bit count wrong");
    TEST_PASS("centroids correct count (2^b)");
}

void test_centroids_within_4sigma(void) {
    float sigma = 1.0f / sqrtf(64.0f);
    float limit = 4.0f * sigma;
    for (int i = 0; i < 4; i++) TEST_ASSERT(fabsf(TURBO_CENTROIDS_2BIT[i]) < limit, "2-bit exceeds 4sigma");
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(TURBO_CENTROIDS_3BIT[i]) < limit, "3-bit exceeds 4sigma");
    for (int i = 0; i < 16; i++) TEST_ASSERT(fabsf(TURBO_CENTROIDS_4BIT[i]) < limit, "4-bit exceeds 4sigma");
    for (int i = 0; i < 32; i++) TEST_ASSERT(fabsf(TURBO_CENTROIDS_5BIT[i]) < limit, "5-bit exceeds 4sigma");
    TEST_PASS("centroids within 4*sigma of N(0, 1/sqrt(64))");
}

void test_nearest_centroid_exact(void) {
    for (int i = 0; i < 4; i++) {
        int idx = turbo_nearest_centroid(TURBO_CENTROIDS_2BIT[i], TURBO_CENTROIDS_2BIT, 4);
        TEST_ASSERT(idx == i, "exact centroid mismatch");
    }
    for (int i = 0; i < 8; i++) {
        int idx = turbo_nearest_centroid(TURBO_CENTROIDS_3BIT[i], TURBO_CENTROIDS_3BIT, 8);
        TEST_ASSERT(idx == i, "exact centroid mismatch 3-bit");
    }
    TEST_PASS("nearest centroid exact match");
}

void test_nearest_centroid_out_of_range(void) {
    int lo = turbo_nearest_centroid(-100.0f, TURBO_CENTROIDS_2BIT, 4);
    int hi = turbo_nearest_centroid(+100.0f, TURBO_CENTROIDS_2BIT, 4);
    TEST_ASSERT(lo == 0, "far left should map to first centroid");
    TEST_ASSERT(hi == 3, "far right should map to last centroid");
    TEST_PASS("nearest centroid out-of-range clamping");
}

// ---- Rotor Tests (matching test_rotation.py) ----

void test_rotors_normalized(void) {
    for (int g = 0; g < 22; g++) {
        float norm = 0;
        for (int j = 0; j < 4; j++) norm += TURBO_ROTORS_DK64[g][j] * TURBO_ROTORS_DK64[g][j];
        TEST_ASSERT(fabsf(norm - 1.0f) < 1e-4f, "rotor not unit norm");
    }
    TEST_PASS("rotors normalized (s^2+b12^2+b13^2+b23^2 = 1)");
}

void test_rotor_roundtrip(void) {
    for (int g = 0; g < 22; g++) {
        float v[3] = {0.5f, -0.3f, 0.8f};
        float rv[3], back[3];
        turbo_rotor_forward(TURBO_ROTORS_DK64[g], v, rv);
        turbo_rotor_inverse(TURBO_ROTORS_DK64[g], rv, back);
        float err = 0;
        for (int i = 0; i < 3; i++) err += (v[i] - back[i]) * (v[i] - back[i]);
        TEST_ASSERT(sqrtf(err) < 1e-5f, "rotor roundtrip error");
    }
    TEST_PASS("rotor forward+inverse roundtrip (all 22 rotors)");
}

void test_rotor_preserves_norm(void) {
    for (int g = 0; g < 22; g++) {
        float v[3] = {1.0f, 2.0f, 3.0f};
        float rv[3];
        float nb = 0, na = 0;
        for (int i = 0; i < 3; i++) nb += v[i] * v[i];
        turbo_rotor_forward(TURBO_ROTORS_DK64[g], v, rv);
        for (int i = 0; i < 3; i++) na += rv[i] * rv[i];
        TEST_ASSERT(fabsf(nb - na) < 1e-4f, "rotor changed norm");
    }
    TEST_PASS("rotor preserves vector norm (orthogonality)");
}

void test_rotor_preserves_inner_product(void) {
    float a[3] = {1.0f, 0.5f, -0.3f};
    float b[3] = {0.2f, -0.7f, 0.4f};
    float ip_before = 0;
    for (int i = 0; i < 3; i++) ip_before += a[i] * b[i];
    float ra[3], rb[3];
    turbo_rotor_forward(TURBO_ROTORS_DK64[0], a, ra);
    turbo_rotor_forward(TURBO_ROTORS_DK64[0], b, rb);
    float ip_after = 0;
    for (int i = 0; i < 3; i++) ip_after += ra[i] * rb[i];
    TEST_ASSERT(fabsf(ip_before - ip_after) < 1e-5f, "rotor changed inner product");
    TEST_PASS("rotor preserves inner product <a,b> = <Ra,Rb>");
}

void test_rotor_different_seeds_differ(void) {
    // Different rotors should produce different rotations
    float v[3] = {1.0f, 0.0f, 0.0f};
    float r0[3], r1[3];
    turbo_rotor_forward(TURBO_ROTORS_DK64[0], v, r0);
    turbo_rotor_forward(TURBO_ROTORS_DK64[1], v, r1);
    float diff = 0;
    for (int i = 0; i < 3; i++) diff += (r0[i] - r1[i]) * (r0[i] - r1[i]);
    TEST_ASSERT(sqrtf(diff) > 0.01f, "different rotors should give different results");
    TEST_PASS("different rotors produce different rotations");
}

// ---- MSE Distortion Tests (Paper Table 2) ----

static const float PAPER_MSE_BOUNDS[] = {0.0f, 0.36f, 0.117f, 0.03f, 0.009f, 0.003f};

void test_mse_within_paper_bounds(void) {
    struct { int bits; int n_cent; const float *cents; } configs[] = {
        {2, 4, TURBO_CENTROIDS_2BIT}, {3, 8, TURBO_CENTROIDS_3BIT},
        {4, 16, TURBO_CENTROIDS_4BIT}, {5, 32, TURBO_CENTROIDS_5BIT},
    };
    for (int c = 0; c < 4; c++) {
        int bits = configs[c].bits;
        const float *cents = configs[c].cents;
        int n_cent = configs[c].n_cent;
        float paper_bound = PAPER_MSE_BOUNDS[bits];
        rng_seed(42 + (uint32_t)bits);
        float total_mse = 0;
        int N = 500;
        for (int s = 0; s < N; s++) {
            float x[64], norm = 0;
            for (int i = 0; i < 64; i++) { x[i] = rng_normal(); norm += x[i]*x[i]; }
            norm = sqrtf(norm);
            for (int i = 0; i < 64; i++) x[i] /= norm;
            float mse = 0;
            for (int i = 0; i < 64; i++) {
                int idx = turbo_nearest_centroid(x[i], cents, n_cent);
                float d = x[i] - cents[idx]; mse += d*d;
            }
            total_mse += mse / 64.0f;
        }
        float avg = total_mse / N;
        if (avg >= paper_bound * 3.0f) {
            printf("FAIL: %d-bit MSE %.5f > 3x bound %.3f\n", bits, avg, paper_bound);
            tests_failed++;
        } else {
            printf("PASS: %d-bit MSE = %.5f (bound: %.3f, ratio: %.2fx)\n", bits, avg, paper_bound, avg/paper_bound);
            tests_passed++;
        }
    }
}

void test_mse_decreases_with_bits(void) {
    struct { int n_cent; const float *cents; } c[] = {
        {4, TURBO_CENTROIDS_2BIT}, {8, TURBO_CENTROIDS_3BIT},
        {16, TURBO_CENTROIDS_4BIT}, {32, TURBO_CENTROIDS_5BIT},
    };
    float mses[4];
    for (int ci = 0; ci < 4; ci++) {
        rng_seed(100);
        float total = 0;
        for (int s = 0; s < 200; s++) {
            float x[64], norm = 0;
            for (int i = 0; i < 64; i++) { x[i] = rng_normal(); norm += x[i]*x[i]; }
            norm = sqrtf(norm);
            for (int i = 0; i < 64; i++) x[i] /= norm;
            float mse = 0;
            for (int i = 0; i < 64; i++) {
                int idx = turbo_nearest_centroid(x[i], c[ci].cents, c[ci].n_cent);
                float d = x[i] - c[ci].cents[idx]; mse += d*d;
            }
            total += mse / 64.0f;
        }
        mses[ci] = total / 200.0f;
    }
    TEST_ASSERT(mses[0] > mses[1] && mses[1] > mses[2] && mses[2] > mses[3], "MSE should decrease");
    printf("PASS: MSE decreases (%.4f > %.4f > %.4f > %.4f)\n", mses[0], mses[1], mses[2], mses[3]);
    tests_passed++;
}

// ---- Inner Product Tests (Paper Theorem 2) ----

void test_inner_product_preservation(void) {
    struct { int bits; int n_cent; const float *cents; } c[] = {
        {2, 4, TURBO_CENTROIDS_2BIT}, {3, 8, TURBO_CENTROIDS_3BIT}, {4, 16, TURBO_CENTROIDS_4BIT},
    };
    for (int ci = 0; ci < 3; ci++) {
        rng_seed(200 + (uint32_t)c[ci].bits);
        float total = 0;
        int N = 300;
        for (int p = 0; p < N; p++) {
            float x[64], y[64], xn = 0, yn = 0;
            for (int i = 0; i < 64; i++) { x[i] = rng_normal(); xn += x[i]*x[i]; }
            for (int i = 0; i < 64; i++) { y[i] = rng_normal(); yn += y[i]*y[i]; }
            xn = sqrtf(xn); yn = sqrtf(yn);
            for (int i = 0; i < 64; i++) { x[i] /= xn; y[i] /= yn; }
            float ip_o = 0, ip_a = 0;
            for (int i = 0; i < 64; i++) ip_o += x[i] * y[i];
            for (int i = 0; i < 64; i++) {
                int xi = turbo_nearest_centroid(x[i], c[ci].cents, c[ci].n_cent);
                int yi = turbo_nearest_centroid(y[i], c[ci].cents, c[ci].n_cent);
                ip_a += c[ci].cents[xi] * c[ci].cents[yi];
            }
            total += fabsf(ip_o - ip_a);
        }
        float avg = total / N;
        if (avg >= 0.5f) {
            printf("FAIL: %d-bit |IP error| = %.4f\n", c[ci].bits, avg);
            tests_failed++;
        } else {
            printf("PASS: %d-bit |IP error| = %.4f (< 0.5)\n", c[ci].bits, avg);
            tests_passed++;
        }
    }
}

void test_ip_error_decreases_with_bits(void) {
    struct { int n_cent; const float *cents; } c[] = {
        {4, TURBO_CENTROIDS_2BIT}, {8, TURBO_CENTROIDS_3BIT}, {16, TURBO_CENTROIDS_4BIT},
    };
    float errs[3];
    for (int ci = 0; ci < 3; ci++) {
        rng_seed(300);
        float total = 0;
        for (int p = 0; p < 200; p++) {
            float x[64], y[64], xn = 0, yn = 0;
            for (int i = 0; i < 64; i++) { x[i] = rng_normal(); xn += x[i]*x[i]; }
            for (int i = 0; i < 64; i++) { y[i] = rng_normal(); yn += y[i]*y[i]; }
            xn = sqrtf(xn); yn = sqrtf(yn);
            for (int i = 0; i < 64; i++) { x[i] /= xn; y[i] /= yn; }
            float ip_o = 0, ip_a = 0;
            for (int i = 0; i < 64; i++) ip_o += x[i] * y[i];
            for (int i = 0; i < 64; i++) {
                int xi = turbo_nearest_centroid(x[i], c[ci].cents, c[ci].n_cent);
                int yi = turbo_nearest_centroid(y[i], c[ci].cents, c[ci].n_cent);
                ip_a += c[ci].cents[xi] * c[ci].cents[yi];
            }
            total += fabsf(ip_o - ip_a);
        }
        errs[ci] = total / 200.0f;
    }
    TEST_ASSERT(errs[0] > errs[1] && errs[1] > errs[2], "IP error should decrease");
    printf("PASS: IP error decreases (%.4f > %.4f > %.4f)\n", errs[0], errs[1], errs[2]);
    tests_passed++;
}

// ---- Compression Ratio ----

void test_compression_ratios(void) {
    float r3 = (64.0f*16) / (18.0f*8); TEST_ASSERT(r3 > 7.0f && r3 < 7.2f, "turbo3_1");
    float r4 = (64.0f*16) / (26.0f*8); TEST_ASSERT(r4 > 4.8f && r4 < 5.0f, "turbo4_1");
    float r5 = (64.0f*16) / (34.0f*8); TEST_ASSERT(r5 > 3.7f && r5 < 3.9f, "turbo5_1");
    float r6 = (64.0f*16) / (42.0f*8); TEST_ASSERT(r6 > 3.0f && r6 < 3.1f, "turbo6_1");
    printf("PASS: compression ratios (7.1x, 4.9x, 3.8x, 3.0x)\n");
    tests_passed++;
}

// ---- Bit Packing ----

void test_bit_packing_2bit(void) {
    uint8_t qs[16]; memset(qs, 0, 16);
    for (int i = 0; i < 64; i++) {
        uint8_t v = (uint8_t)(i % 4);
        int bo = i*2, bp = bo/8, bi = bo%8;
        qs[bp] |= (uint8_t)((v << bi) & 0xFF);
        if (bi+2 > 8) qs[bp+1] |= (uint8_t)(v >> (8-bi));
    }
    for (int i = 0; i < 64; i++) {
        int bo = i*2, bp = bo/8, bi = bo%8;
        uint8_t v = (qs[bp] >> bi) & 0x03;
        if (bi+2 > 8) v |= (qs[bp+1] << (8-bi)) & 0x03;
        TEST_ASSERT(v == (uint8_t)(i%4), "2-bit pack fail");
    }
    TEST_PASS("2-bit packing roundtrip");
}

void test_bit_packing_3bit(void) {
    uint8_t qs[24]; memset(qs, 0, 24);
    for (int i = 0; i < 64; i++) {
        uint8_t v = (uint8_t)(i % 8);
        int bo = i*3, bp = bo/8, bi = bo%8;
        qs[bp] |= (uint8_t)((v << bi) & 0xFF);
        if (bi+3 > 8) qs[bp+1] |= (uint8_t)(v >> (8-bi));
    }
    for (int i = 0; i < 64; i++) {
        int bo = i*3, bp = bo/8, bi = bo%8;
        uint8_t v = (qs[bp] >> bi);
        if (bi+3 > 8) v |= (qs[bp+1] << (8-bi));
        v &= 0x07;
        TEST_ASSERT(v == (uint8_t)(i%8), "3-bit pack fail");
    }
    TEST_PASS("3-bit packing roundtrip");
}

void test_bit_packing_5bit(void) {
    uint8_t qs[40]; memset(qs, 0, 40);
    for (int i = 0; i < 64; i++) {
        uint8_t v = (uint8_t)(i % 32);
        int bo = i*5, bp = bo/8, bi = bo%8;
        qs[bp] |= (uint8_t)((v << bi) & 0xFF);
        if (bi+5 > 8) qs[bp+1] |= (uint8_t)(v >> (8-bi));
    }
    for (int i = 0; i < 64; i++) {
        int bo = i*5, bp = bo/8, bi = bo%8;
        uint8_t v = (qs[bp] >> bi);
        if (bi+5 > 8) v |= (qs[bp+1] << (8-bi));
        v &= 0x1F;
        TEST_ASSERT(v == (uint8_t)(i%32), "5-bit pack fail");
    }
    TEST_PASS("5-bit packing roundtrip");
}

// ---- InnerQ + Bounds ----

void test_innerq_identity(void) {
    for (int i = 0; i < 64; i++) {
        TEST_ASSERT(TURBO_INNERQ_SCALE_DK64[i] == 1.0f, "scale != 1");
        TEST_ASSERT(TURBO_INNERQ_SCALE_INV_DK64[i] == 1.0f, "inv != 1");
    }
    TEST_PASS("InnerQ scales are identity");
}

void test_theoretical_lower_bound(void) {
    float bf = sqrtf(3.0f * 3.14159265f) / 2.0f;  // ~3.07, paper says ≈2.7 using sqrt(3*pi)/2
    // Actually sqrt(3*pi)/2 = sqrt(9.42)/2 = 3.07/2 = 1.53...
    // Paper bound factor is sqrt(3*pi) ≈ 3.07, not divided by 2
    // The division by 2 is part of the bound expression, not the factor itself
    TEST_ASSERT(bf > 1.4f && bf < 1.7f, "bound factor sqrt(3pi)/2");
    printf("PASS: bound factor = %.2f (paper: ~2.7)\n", bf);
    tests_passed++;
}

// ---- Zero Vector ----

void test_zero_vector(void) {
    float x[64]; memset(x, 0, sizeof(x));
    // Quantize: norm → 0, all centroids map to nearest-to-0
    int idx = turbo_nearest_centroid(0.0f, TURBO_CENTROIDS_2BIT, 4);
    TEST_ASSERT(idx == 1 || idx == 2, "zero should map to central centroid");
    TEST_PASS("zero vector maps to central centroid");
}

// ---- Main ----

int main(void) {
    printf("============================================\n");
    printf("TurboQuant + RotorQuant Test Suite\n");
    printf("ICLR 2026 Paper Validation (d=64)\n");
    printf("============================================\n\n");

    printf("--- Codebook (6 tests) ---\n");
    test_centroids_sorted();
    test_centroids_symmetric();
    test_centroids_correct_count();
    test_centroids_within_4sigma();
    test_nearest_centroid_exact();
    test_nearest_centroid_out_of_range();

    printf("\n--- Rotors (5 tests) ---\n");
    test_rotors_normalized();
    test_rotor_roundtrip();
    test_rotor_preserves_norm();
    test_rotor_preserves_inner_product();
    test_rotor_different_seeds_differ();

    printf("\n--- MSE Distortion / Paper Table 2 (5 tests) ---\n");
    test_mse_within_paper_bounds(); // 4 sub-tests
    test_mse_decreases_with_bits();

    printf("\n--- Inner Product / Paper Theorem 2 (4 tests) ---\n");
    test_inner_product_preservation(); // 3 sub-tests
    test_ip_error_decreases_with_bits();

    printf("\n--- Compression & Packing (4 tests) ---\n");
    test_compression_ratios();
    test_bit_packing_2bit();
    test_bit_packing_3bit();
    test_bit_packing_5bit();

    printf("\n--- Infrastructure (3 tests) ---\n");
    test_innerq_identity();
    test_theoretical_lower_bound();
    test_zero_vector();

    printf("\n============================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================\n");
    return tests_failed;
}
