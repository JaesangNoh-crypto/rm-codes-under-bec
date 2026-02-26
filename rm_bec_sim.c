/*
 * rm_bec_sim.c
 *
 * High-performance Monte Carlo simulation of P(ambiguous) for bit 0
 * of punctured Reed-Muller codes RM*(r,m) on Binary Erasure Channel.
 * Code length n = 2^m - 1 (evaluation at nonzero points 1..2^m-1).
 *
 * Optimizations:
 *   1. AVX2 SIMD: 256-bit packed XOR (4x throughput vs scalar 64-bit)
 *   2. On-the-fly H columns: stores only nk monomial masks (uint32_t)
 *      instead of full n × nw matrix — huge memory saving for large m
 *   3. No back-reduction: echelon form insert costs ~50% less,
 *      span check correctness preserved (proven for row echelon form)
 *   - Incremental echelon basis (no full matrix copy per trial)
 *   - OpenMP parallel trials with thread-local workspace
 *   - xoshiro256** PRNG per thread (no locks, no false sharing)
 *   - Geometric random variate for fast erasure pattern generation
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -o rm_bec_sim rm_bec_sim.c -lm
 *
 * Usage:
 *   ./rm_bec_sim -r <order> -m <param> -f <nframes> [-s start] [-e end] [-d step] [-t threads] [-o file.csv]
 *
 * Examples:
 *   ./rm_bec_sim -r 1 -m 10 -f 100000
 *   ./rm_bec_sim -r 2 -m 8 -f 50000 -s 0.35 -e 0.65 -d 0.005 -t 8
 *   ./rm_bec_sim -r 3 -m 10 -f 50000 -o results.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <getopt.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ================================================================
 *  [OPT-1] Vectorized GF(2) XOR: AVX2 (256-bit) with scalar tail
 * ================================================================ */
static inline void xor_vec(uint64_t *dst, const uint64_t *src, int nw) {
    int i = 0;
#ifdef __AVX2__
    for (; i + 4 <= nw; i += 4) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(dst + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_xor_si256(a, b));
    }
#endif
    for (; i < nw; i++) dst[i] ^= src[i];
}

/* ================================================================
 *  xoshiro256** PRNG
 * ================================================================ */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3], 45);
    return res;
}

static inline double rng_double(rng_t *r) {
    return (rng_next(r) >> 11) * 0x1.0p-53;
}

static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* ================================================================
 *  [OPT-2] H matrix: hybrid mode
 *
 *  Small H (< 1MB):  materialize full column-major packed H matrix
 *  Large H (>= 1MB): on-the-fly via nk monomial masks (uint32_t)
 *    H[row i][col j] = 1  iff  ((j+1) & masks[i]) == masks[i]
 *
 *  Memory comparison for RM(1,15):
 *    Stored:     n × nw × 8 = 32768 × 512 × 8 = 128 MB
 *    On-the-fly: nk × 4     = 32752 × 4       = 128 KB  (1000x saving)
 * ================================================================ */
typedef struct {
    int       n;        /* code length = 2^m - 1 (punctured)  */
    int       k;        /* dimension = rm_dim(r, m)            */
    int       nk;       /* # monomial rows of H                */
    int       nw;       /* ceil(nk / 64)                       */
    uint64_t *data;     /* materialized H, NULL if on-the-fly  */
    uint32_t *masks;    /* monomial masks,  NULL if stored     */
} HMat;

static HMat *hmat_alloc(int n, int nk) {
    HMat *H = calloc(1, sizeof *H);
    H->n = n; H->nk = nk; H->nw = (nk + 63) >> 6;
    return H;
}

static void hmat_free(HMat *H) {
    free(H->data); free(H->masks); free(H);
}

/* Get column j: stored → direct pointer, on-the-fly → compute into buf */
static inline const uint64_t *hmat_col(const HMat *H, int j, uint64_t *buf) {
    if (H->data)
        return H->data + (size_t)j * H->nw;

    /* On-the-fly: evaluate all monomials at point j+1 (nonzero points) */
    memset(buf, 0, H->nw * sizeof(uint64_t));
    const uint32_t jj = (uint32_t)(j + 1);
    const uint32_t *m = H->masks;
    const int nk = H->nk;
    for (int i = 0; i < nk; i++)
        if ((jj & m[i]) == m[i])
            buf[i >> 6] |= 1ULL << (i & 63);
    return buf;
}

/* ================================================================
 *  Reed-Muller code construction — hybrid mode
 *
 *  RM*(r,m): n = 2^m - 1, k = rm_dim(r, m)
 *  Parity check matrix H = generator of dual code RM(m-r-1, m)
 *  Rows = monomials of degree <= m-r-1, evaluated at nonzero points 1..2^m-1
 *
 *  Hybrid strategy:
 *    H < H_CACHE_THRESH → materialize full H (fast pointer access)
 *    H >= H_CACHE_THRESH → on-the-fly via monomial masks (low memory)
 * ================================================================ */
#define H_CACHE_THRESH  (1 << 20)   /* 1 MB: L2 cache boundary */

static int binom(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    int r = 1;
    for (int i = 0; i < k; i++)
        r = r * (n - i) / (i + 1);
    return r;
}

static int rm_dim(int r, int m) {
    int d = 0;
    for (int i = 0; i <= r && i <= m; i++)
        d += binom(m, i);
    return d;
}

/* Build monomial masks for RM(r,m) dual code */
static uint32_t *build_masks(int m, int dual_r, int nk) {
    uint32_t *masks = malloc(nk * sizeof(uint32_t));
    int row = 0;
    for (int deg = 0; deg <= dual_r; deg++) {
        if (deg == 0) {
            masks[row++] = 0;
            continue;
        }
        uint32_t mask  = (1u << deg) - 1;
        uint32_t limit = 1u << m;
        while (mask < limit) {
            masks[row++] = mask;
            uint32_t c  = mask & (-mask);
            uint32_t rr = mask + c;
            mask = (((rr ^ mask) >> 2) / c) | rr;
        }
    }
    return masks;
}

/* Materialize full H matrix from masks */
static void materialize_h(HMat *H, const uint32_t *masks) {
    int n = H->n, nk = H->nk;
    H->data = calloc((size_t)n * H->nw, sizeof(uint64_t));
    for (int j = 0; j < n; j++) {
        uint64_t *col = H->data + (size_t)j * H->nw;
        uint32_t jj = (uint32_t)(j + 1);   /* evaluate at nonzero point */
        for (int i = 0; i < nk; i++)
            if ((jj & masks[i]) == masks[i])
                col[i >> 6] |= 1ULL << (i & 63);
    }
}

static HMat *build_rm_hmat(int r, int m) {
    if (r < 0 || m < 1 || r >= m) {
        fprintf(stderr, "RM(%d,%d): need 0 <= r < m for non-trivial code.\n", r, m);
        return NULL;
    }

    int n  = (1 << m) - 1;
    int k  = rm_dim(r, m);
    int dual_r = m - r - 1;
    int nk = rm_dim(dual_r, m);   /* # monomial rows */

    HMat *H = hmat_alloc(n, nk);
    H->k = k;

    /* Always build masks first */
    uint32_t *masks = build_masks(m, dual_r, nk);
    if (!masks) { hmat_free(H); return NULL; }

    size_t h_bytes = (size_t)n * H->nw * sizeof(uint64_t);

    if (h_bytes < H_CACHE_THRESH) {
        /* Small H: materialize for fast pointer access */
        materialize_h(H, masks);
        free(masks);
        printf("Built RM*(%d,%d): (%d, %d), R = %.4f, H: %d x %d, %d words/col\n",
               r, m, n, k, (double)k / n, nk, n, H->nw);
        printf("  Stored mode: %zu bytes (fits in cache)\n", h_bytes);
    } else {
        /* Large H: keep masks for on-the-fly computation */
        H->masks = masks;
        size_t mask_bytes = (size_t)nk * sizeof(uint32_t);
        printf("Built RM*(%d,%d): (%d, %d), R = %.4f, H: %d x %d, %d words/col\n",
               r, m, n, k, (double)k / n, nk, n, H->nw);
        printf("  On-the-fly mode: %zu B masks vs %zu B full H (%.0fx saving)\n",
               mask_bytes, h_bytes, (double)h_bytes / mask_bytes);
    }
    return H;
}

/* ================================================================
 *  [OPT-3] GF(2) Incremental Echelon Basis — NO back-reduction
 *
 *  Row echelon form (not reduced):
 *    basis[i] has bit piv[i] set, and NO bits at piv[j] for j < i
 *    (guaranteed by forward reduction during insertion)
 *    basis[i] MAY have bits at piv[j] for j > i (not back-reduced)
 *
 *  Correctness of span check:
 *    Forward sweep clears each pivot bit exactly once.
 *    After sweep, t=0 iff v ∈ span(B).
 *    (Each basis vector b_j with j>i cannot have bit p_i,
 *     so clearing p_i at step i is permanent.)
 *
 *  Benefit: saves O(rank × nw) work per insert (~50% of insert cost)
 * ================================================================ */
typedef struct {
    int       nw, nk;
    int       rank;
    int       cap;
    uint64_t *pool;     /* basis vectors: basis[i] at pool + i*nw */
    int      *piv;      /* pivot bit index per basis vector       */
    uint64_t *tmp;      /* scratch for reduction                  */
    uint64_t *col_buf;  /* scratch for on-the-fly column compute  */
} GEWork;

static GEWork *ge_alloc(int nk, int nw, int cap) {
    GEWork *w = malloc(sizeof *w);
    w->nw = nw; w->nk = nk; w->rank = 0; w->cap = cap;
    w->pool    = malloc((size_t)cap * nw * sizeof(uint64_t));
    w->piv     = malloc(cap * sizeof(int));
    w->tmp     = malloc(nw * sizeof(uint64_t));
    w->col_buf = malloc(nw * sizeof(uint64_t));
    return w;
}

static void ge_free(GEWork *w) {
    free(w->pool); free(w->piv); free(w->tmp); free(w->col_buf); free(w);
}

static inline void ge_reset(GEWork *w) { w->rank = 0; }

static inline uint64_t *ge_basis(GEWork *w, int i) {
    return w->pool + (size_t)i * w->nw;
}

/* Insert column vector into echelon basis (no back-reduction). */
static void ge_insert(GEWork *w, const uint64_t *v) {
    int nw = w->nw;
    memcpy(w->tmp, v, nw * sizeof(uint64_t));

    /* Forward reduce against existing basis */
    for (int i = 0; i < w->rank; i++) {
        int pb = w->piv[i];
        if ((w->tmp[pb >> 6] >> (pb & 63)) & 1)
            xor_vec(w->tmp, ge_basis(w, i), nw);
    }

    /* Find pivot (lowest set bit) */
    int p = -1;
    for (int u = 0; u < nw && p < 0; u++)
        if (w->tmp[u]) p = (u << 6) | __builtin_ctzll(w->tmp[u]);
    if (p < 0 || p >= w->nk) return;

    /* Store — no back-reduction */
    memcpy(ge_basis(w, w->rank), w->tmp, nw * sizeof(uint64_t));
    w->piv[w->rank] = p;
    w->rank++;
}

/* Check if v ∈ span(basis) via forward reduction */
static inline int ge_in_span(const GEWork *w, const uint64_t *v) {
    int nw = w->nw;
    uint64_t *t = w->tmp;
    memcpy(t, v, nw * sizeof(uint64_t));
    for (int i = 0; i < w->rank; i++) {
        int pb = w->piv[i];
        if ((t[pb >> 6] >> (pb & 63)) & 1)
            xor_vec(t, w->pool + (size_t)i * nw, nw);
    }
    for (int u = 0; u < nw; u++) if (t[u]) return 0;
    return 1;
}

/* ================================================================
 *  Simulation
 * ================================================================ */
static double simulate_eps(const HMat *H, double eps, long long nframes, int nthr, int report_progress) {
    long long tot_amb = 0;
    long long done_counter = 0;
    long long report_interval = nframes / 100;
    if (report_interval < 1) report_interval = 1;
    const double log1meps = (eps < 1.0 - 1e-15) ? log(1.0 - eps) : -40.0;

    #pragma omp parallel num_threads(nthr) reduction(+:tot_amb)
    {
        int tid = omp_get_thread_num();
        rng_t rng;
        rng_seed(&rng, 42ULL + (uint64_t)tid * 1000003ULL +
                 (uint64_t)(eps * 1e9));
        GEWork *ws = ge_alloc(H->nk, H->nw, H->nk);
        long long local_done = 0;

        #pragma omp for schedule(dynamic, 4096)
        for (long long f = 0; f < nframes; f++) {
            ge_reset(ws);

            if (eps < 1e-15) {
                /* No erasures besides bit 0 */
            } else if (eps > 1.0 - 1e-15) {
                /* All erased */
                for (int j = 1; j < H->n; j++) {
                    const uint64_t *col = hmat_col(H, j, ws->col_buf);
                    ge_insert(ws, col);
                }
            } else {
                if (eps < 0.35) {
                    /* Geometric skip for sparse erasures */
                    int j = 1;
                    while (j < H->n && ws->rank < ws->nk) {
                        double u = rng_double(&rng);
                        if (u < 1e-300) u = 1e-300;
                        int skip = (int)(log(u) / log1meps);
                        j += skip;
                        if (j < H->n) {
                            const uint64_t *col = hmat_col(H, j, ws->col_buf);
                            ge_insert(ws, col);
                            j++;
                        }
                    }
                } else {
                    for (int j = 1; j < H->n && ws->rank < ws->nk; j++) {
                        if (rng_double(&rng) < eps) {
                            const uint64_t *col = hmat_col(H, j, ws->col_buf);
                            ge_insert(ws, col);
                        }
                    }
                }
            }

            const uint64_t *col0 = hmat_col(H, 0, ws->col_buf);
            if (ge_in_span(ws, col0))
                tot_amb++;

            if (report_progress && (++local_done & 1023) == 0) {
                #pragma omp atomic
                done_counter += 1024;
                long long d = done_counter; /* racy read is fine for progress */
                if (d / report_interval != (d - 1024) / report_interval) {
                    fprintf(stderr, "PROGRESS %lld %lld\n", d, nframes);
                    fflush(stderr);
                }
            }
        }
        if (report_progress && (local_done & 1023)) {
            #pragma omp atomic
            done_counter += (local_done & 1023);
        }
        ge_free(ws);
    }
    if (report_progress) {
        fprintf(stderr, "PROGRESS %lld %lld\n", nframes, nframes);
        fflush(stderr);
    }
    return (double)tot_amb / nframes;
}

/* ================================================================
 *  Main
 * ================================================================ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -r <order> -m <param> -f <nframes> [-s start] [-e end] [-d step] [-t threads] [-o file.csv] [-p]\n"
        "\nOptions:\n"
        "  -r order     Reed-Muller order r (0 <= r < m)\n"
        "  -m param     Reed-Muller parameter m (code length = 2^m - 1)\n"
        "  -f frames    Monte Carlo frames per epsilon point\n"
        "  -s start     Epsilon range start (default 0.4)\n"
        "  -e end       Epsilon range end   (default 0.5)\n"
        "  -d step      Epsilon step        (default 0.02)\n"
        "  -t threads   OpenMP threads (default: all)\n"
        "  -o file.csv  Write CSV output\n"
        "  -p           Report frame progress to stderr\n",
        prog);
}

int main(int argc, char **argv) {
    int    rm_r    = -1;
    int    rm_m    = -1;
    char  *csvfile = NULL;
    long long nframes = 0;
    double eps_s = 0.4, eps_e = 0.5, eps_d = 0.02;
    int    nthr = omp_get_max_threads();
    int    report_progress = 0;

    int opt;
    while ((opt = getopt(argc, argv, "r:m:f:s:e:d:t:o:ph")) != -1) {
        switch (opt) {
        case 'r': rm_r    = atoi(optarg);  break;
        case 'm': rm_m    = atoi(optarg);  break;
        case 'f': nframes = atoll(optarg);  break;
        case 's': eps_s   = atof(optarg);  break;
        case 'e': eps_e   = atof(optarg);  break;
        case 'd': eps_d   = atof(optarg);  break;
        case 't': nthr    = atoi(optarg);  break;
        case 'o': csvfile = optarg;        break;
        case 'p': report_progress = 1;     break;
        default: usage(argv[0]); return 1;
        }
    }
    if (rm_r < 0 || rm_m < 0 || nframes <= 0) { usage(argv[0]); return 1; }

    /* Build H */
    printf("Building RM(%d,%d) code ...\n", rm_r, rm_m);
    HMat *H = build_rm_hmat(rm_r, rm_m);
    if (!H) { fprintf(stderr, "Failed to obtain H matrix.\n"); return 1; }

    int n = H->n, k = H->k;
    printf("Code: (%d, %d), R = %.4f, %d words/col, %d threads\n",
           n, k, (double)k/n, H->nw, nthr);
    printf("Frames: %lld, eps: [%.4f, %.4f] step %.4f\n", nframes, eps_s, eps_e, eps_d);
#ifdef __AVX2__
    printf("SIMD: AVX2 enabled (256-bit XOR)\n\n");
#else
    printf("SIMD: scalar fallback (64-bit XOR)\n\n");
#endif

    FILE *csv = NULL;
    if (csvfile) {
        csv = fopen(csvfile, "w");
        if (csv) fprintf(csv, "epsilon,P_ambiguous,frames,time_sec\n");
    }

    printf("%-12s %-18s %-14s %s\n", "epsilon", "P(ambiguous)", "frames", "time");
    printf("%-12s %-18s %-14s %s\n", "--------", "----------", "------", "----");

    for (double eps = eps_s; eps <= eps_e + eps_d * 0.01; eps += eps_d) {
        if (eps > 1.0) eps = 1.0;
        double t0 = omp_get_wtime();
        double pamb = simulate_eps(H, eps, nframes, nthr, report_progress);
        double dt = omp_get_wtime() - t0;

        printf("%-12.4f %-18.8f %-14lld %.3fs\n", eps, pamb, nframes, dt);
        fflush(stdout);

        if (csv) {
            fprintf(csv, "%.6f,%.10f,%lld,%.4f\n", eps, pamb, nframes, dt);
            fflush(csv);
        }
        if (eps >= 1.0 - 1e-9) break;
    }

    if (csv) fclose(csv);
    hmat_free(H);
    printf("\nDone.\n");
    return 0;
}
