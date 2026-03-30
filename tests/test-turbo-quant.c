#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// Include turbo constants and rotor functions
#include "ggml-turbo-rotors.h"

int main(void) {
    int pass = 0, fail = 0;

    // Test 1: Centroids sorted (strictly increasing)
    {
        int ok = 1;
        for (int i = 1; i < 4; i++)  { if (!(TURBO_CENTROIDS_2BIT[i] > TURBO_CENTROIDS_2BIT[i-1])) ok = 0; }
        for (int i = 1; i < 8; i++)  { if (!(TURBO_CENTROIDS_3BIT[i] > TURBO_CENTROIDS_3BIT[i-1])) ok = 0; }
        for (int i = 1; i < 16; i++) { if (!(TURBO_CENTROIDS_4BIT[i] > TURBO_CENTROIDS_4BIT[i-1])) ok = 0; }
        for (int i = 1; i < 32; i++) { if (!(TURBO_CENTROIDS_5BIT[i] > TURBO_CENTROIDS_5BIT[i-1])) ok = 0; }
        if (ok) { printf("PASS: centroids sorted\n"); pass++; }
        else    { printf("FAIL: centroids sorted\n"); fail++; }
    }

    // Test 2: Centroids symmetric (c[i] == -c[n-1-i])
    {
        int ok = 1;
        for (int i = 0; i < 2; i++) {
            if (fabsf(TURBO_CENTROIDS_2BIT[i] + TURBO_CENTROIDS_2BIT[3-i]) > 1e-6f) ok = 0;
        }
        for (int i = 0; i < 4; i++) {
            if (fabsf(TURBO_CENTROIDS_3BIT[i] + TURBO_CENTROIDS_3BIT[7-i]) > 1e-6f) ok = 0;
        }
        for (int i = 0; i < 8; i++) {
            if (fabsf(TURBO_CENTROIDS_4BIT[i] + TURBO_CENTROIDS_4BIT[15-i]) > 1e-6f) ok = 0;
        }
        for (int i = 0; i < 16; i++) {
            if (fabsf(TURBO_CENTROIDS_5BIT[i] + TURBO_CENTROIDS_5BIT[31-i]) > 1e-6f) ok = 0;
        }
        if (ok) { printf("PASS: centroids symmetric\n"); pass++; }
        else    { printf("FAIL: centroids symmetric\n"); fail++; }
    }

    // Test 3: Rotor normalization (s^2 + b12^2 + b13^2 + b23^2 == 1)
    {
        int ok = 1;
        for (int g = 0; g < 22; g++) {
            float norm = 0.0f;
            for (int j = 0; j < 4; j++) norm += TURBO_ROTORS_DK64[g][j] * TURBO_ROTORS_DK64[g][j];
            if (fabsf(norm - 1.0f) > 1e-4f) {
                printf("  rotor %d: norm = %.6f\n", g, norm);
                ok = 0;
            }
        }
        if (ok) { printf("PASS: rotors normalized\n"); pass++; }
        else    { printf("FAIL: rotors normalized\n"); fail++; }
    }

    // Test 4: Rotor forward + inverse roundtrip
    {
        float v[3] = {0.5f, -0.3f, 0.8f};
        int ok = 1;
        for (int g = 0; g < 22; g++) {
            float rv[3], back[3];
            turbo_rotor_forward(TURBO_ROTORS_DK64[g], v, rv);
            turbo_rotor_inverse(TURBO_ROTORS_DK64[g], rv, back);
            float err = 0.0f;
            for (int i = 0; i < 3; i++) err += (v[i] - back[i]) * (v[i] - back[i]);
            if (sqrtf(err) > 1e-5f) {
                printf("  rotor %d: roundtrip err = %.2e\n", g, sqrtf(err));
                ok = 0;
            }
        }
        if (ok) { printf("PASS: rotor roundtrip (all 22 rotors)\n"); pass++; }
        else    { printf("FAIL: rotor roundtrip\n"); fail++; }
    }

    // Test 5: Nearest centroid finds exact match
    {
        int ok = 1;
        for (int i = 0; i < 8; i++) {
            int idx = turbo_nearest_centroid(TURBO_CENTROIDS_3BIT[i], TURBO_CENTROIDS_3BIT, 8);
            if (idx != i) {
                printf("  centroid %d: expected idx %d, got %d\n", i, i, idx);
                ok = 0;
            }
        }
        if (ok) { printf("PASS: nearest centroid exact match\n"); pass++; }
        else    { printf("FAIL: nearest centroid exact match\n"); fail++; }
    }

    // Test 6: InnerQ scale arrays are all 1.0 (identity default)
    {
        int ok = 1;
        for (int i = 0; i < 64; i++) {
            if (TURBO_INNERQ_SCALE_DK64[i] != 1.0f || TURBO_INNERQ_SCALE_INV_DK64[i] != 1.0f) ok = 0;
        }
        if (ok) { printf("PASS: InnerQ scales are identity\n"); pass++; }
        else    { printf("FAIL: InnerQ scales are identity\n"); fail++; }
    }

    // Test 7: Rotor preserves vector norm (rotation is orthogonal)
    {
        float v[3] = {1.0f, 2.0f, 3.0f};
        float v_norm2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        int ok = 1;
        for (int g = 0; g < 22; g++) {
            float rv[3];
            turbo_rotor_forward(TURBO_ROTORS_DK64[g], v, rv);
            float rv_norm2 = rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2];
            if (fabsf(rv_norm2 - v_norm2) > 1e-4f) {
                printf("  rotor %d: |v|^2=%.4f, |Rv|^2=%.4f\n", g, v_norm2, rv_norm2);
                ok = 0;
            }
        }
        if (ok) { printf("PASS: rotor preserves norm\n"); pass++; }
        else    { printf("FAIL: rotor preserves norm\n"); fail++; }
    }

    printf("\nResults: %d passed, %d failed\n", pass, fail);
    return fail;
}
