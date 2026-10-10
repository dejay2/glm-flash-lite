// N135 AVX-512 (Zen 4/5, Ice Lake+) p16 I16 kernel. Included from ft_n135.h INSIDE namespace n135 (not standalone).
//
// Same integer sums as the AVX2 p16 kernel (tile_accum_p16 / rect<M, 3>), so with the same unit shapes the result is
// bit-identical to it (verified by the bench: check=1 prints "differ" counts, and by tests in bench/cpu_kernel):
//  * decode: one vpermb (64 stream bytes -> 8 x 8-byte big-endian qwords) + one vpmultishiftqb (bit-granular byte extract)
//    yield 32 16-bit trellis states per zmm, replacing the AVX2 two-shuffle / srlv / sllv / blend sequence. A qword holds
//    the states of weights (g, j), (g+1, j), (g, j+1), (g+1, j+1), i.e. two dword pairs exactly as the AVX2 madd expects;
//  * hash + byte sum: the same 16-bit mullo / mulhi / mullo + two vpmaddubsw as pair_sums();
//  * accumulate: vpdpwssd (AVX512_VNNI) against the unchanged int16 activation layout (one 64 B row = X[pair 0][16], X[pair 1][16]);
//  * the 8 accumulators of a tile stay in registers over each 8-row scale block (tile-major, like rect_rt1); at the flush the
//    two 8-lane halves of a zmm (pair 0 / pair 1) are added in int32 (exact), then converted and fma'd with the block scale
//    exactly like rect<M, 3>.
// Selected at runtime (cpu_has_512 + GLM53_NV_CPU_ISA); the AVX2 path is unchanged and stays the fallback.
#pragma once

inline int g_isa512 = -1;   // -1 unset, 0 = AVX2 kernels, 1 = AVX-512 p16 kernel

inline bool cpu_has_512()
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512vnni");
}
inline bool use512() { return g_isa512 > 0; }

// which 64-byte window of the 96-byte tile serves zmm c: 0 = tile[0..64), 1 = tile[32..96), 2 = tile[64..96) ++ tile[0..32) (wrap)
constexpr int WK512[8] = { 2, 0, 0, 0, 1, 1, 1, 2 };
alignas(64) inline uint8_t IDX512[8][64];
alignas(64) inline uint8_t CTL512[64];

inline void init_512()
{
    for (int c = 0; c < 8; ++c)
        for (int h = 0; h < 2; ++h)
            for (int u = 0; u < 4; ++u)
            {
                const int g0 = 4 * c + 2 * h, ta = 8 * g0 + 2 * u;
                int b0 = 3 * ta - 13;
                if (b0 < 0) b0 += 768;
                const int B = b0 / 8, e = b0 % 8, qw = h * 4 + u;
                for (int k = 0; k < 8; ++k)
                {
                    const int mm = mem_of_stream((B + 7 - k) % 96);
                    int pos = -1;
                    if (WK512[c] == 0) pos = mm < 64 ? mm : -1;
                    else if (WK512[c] == 1) pos = mm >= 32 ? mm - 32 : -1;
                    else pos = mm >= 64 ? mm - 64 : (mm < 32 ? 32 + mm : -1);
                    if (pos < 0) { fprintf(stderr, "n135 avx512 window error c=%d h=%d u=%d k=%d\n", c, h, u, k); exit(1); }
                    IDX512[c][qw * 8 + k] = uint8_t(pos);
                }
                // output words: (g, j), (g+1, j), (g, j+1), (g+1, j+1); state = (V >> sh) & 0xffff, V = 8 stream bytes big-endian
                const int dd[4] = { 0, 24, 3, 27 };
                for (int w = 0; w < 4; ++w)
                {
                    const int sh = 48 - (e + dd[w]);
                    if (sh < 0 || sh > 48) { fprintf(stderr, "n135 avx512 shift error\n"); exit(1); }
                    CTL512[qw * 8 + 2 * w] = uint8_t(sh);
                    CTL512[qw * 8 + 2 * w + 1] = uint8_t(sh + 8);
                }
            }
}

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512vnni")

struct K512
{
    __m512i ctl, c1, c2, one8;
};
inline K512 k512()
{
    return { _mm512_load_si512(CTL512), _mm512_set1_epi16(short(MUL1 & 0xffff)), _mm512_set1_epi16(short(MUL1 >> 16)), _mm512_set1_epi8(1) };
}

// 32 weights of zmm c (pair sums of groups 4c, 4c+1 in the low 8 dwords, 4c+2, 4c+3 in the high 8), int16 per weight
template <int C>
__attribute__((always_inline)) inline __m512i dec512(const __m512i* win, const K512& K)
{
    const __m512i src = win[WK512[C]];
    const __m512i st = _mm512_multishift_epi64_epi8(K.ctl, _mm512_permutexvar_epi8(_mm512_load_si512(IDX512[C]), src));
    const __m512i lo = _mm512_mullo_epi16(st, K.c1);
    const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(st, K.c1), _mm512_mullo_epi16(st, K.c2));
    return _mm512_add_epi16(_mm512_maddubs_epi16(lo, K.one8), _mm512_maddubs_epi16(hi, K.one8));
}

template <int M, int C = 0>
__attribute__((always_inline)) inline void tile512(const __m512i* win, const __m512i* x, __m512i* a, const K512& K)
{
    if constexpr (C < 8)
    {
        const __m512i s = dec512<C>(win, K);
        #pragma GCC unroll 2
        for (int i = 0; i < M; ++i) a[C * M + i] = _mm512_dpwssd_epi32(a[C * M + i], s, x[i]);
        tile512<M, C + 1>(win, x, a, K);
    }
}

// tile rows [r0, r0 + R) (multiples of 8) x tiles [t0, t0 + C) of `mat`, M <= 2 tokens (int16 p16 activations xin[i], block scales qs[i]).
// Writes the lane-reduced raw dot products y[i * ldy + t * 16 + c], c in 0..15, like rect<M, 3>.
template <int M>
void rect512(const Mat& mat, int r0, int R, int t0, int C, const void* const* xin, const float* const* qs, float* y, size_t ldy)
{
    static thread_local float* accf = nullptr;
    if (!accf) accf = static_cast<float*>(std::aligned_alloc(64, size_t(MAXCT) * 64 * 8 * 4));
    std::memset(accf, 0, size_t(C) * 64 * M * 4);
    const K512 K = k512();
    const int tiles_n = mat.n / 16, tk = mat.k / 16;
    const size_t row_stride = mat.swz ? size_t(8) * TILE_BYTES : size_t(tiles_n) * TILE_BYTES;
    const uint8_t* base = mat.swz ? mat.tr + size_t(t0 / 8) * tk * 8 * TILE_BYTES : mat.tr + size_t(t0) * TILE_BYTES;
    const int16_t* xq[M];
    for (int i = 0; i < M; ++i) xq[i] = static_cast<const int16_t*>(xin[i]);
    const int lines = (C * TILE_BYTES + 63) / 64;
    const int pfr = K_.pfr;
    const int PFG = std::max(1, K_.pfg);
    if (K_.pfpro) for (int r = r0; r < std::min(r0 + 8, r0 + R); ++r) for (int l = 0; l < lines; ++l) _mm_prefetch(reinterpret_cast<const char*>(base + size_t(r) * row_stride) + l * 64, _MM_HINT_T0);
    for (int g8 = r0; g8 < r0 + R; g8 += 8)
    {
        __m256 sc[M];
        for (int i = 0; i < M; ++i) sc[i] = _mm256_set1_ps(qs[i][g8 >> 3]);
        const bool pfn = pfr > 0 && g8 + 8 < tk;
        for (int t = 0; t < C; ++t)
        {
            __m512i a[8 * M];
            #pragma GCC unroll 16
            for (int k = 0; k < 8 * M; ++k) a[k] = _mm512_setzero_si512();
            const uint8_t* p = base + size_t(g8) * row_stride + size_t(t) * TILE_BYTES;
            #pragma GCC unroll 1
            for (int rr = 0; rr < 8; ++rr)
            {
                const uint8_t* tl = p + size_t(rr) * row_stride;
                __m512i win[3];
                win[0] = _mm512_loadu_si512(tl);
                win[1] = _mm512_loadu_si512(tl + 32);
                win[2] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tl + 64))),
                                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tl)), 1);
                __m512i x[M];
                for (int i = 0; i < M; ++i) x[i] = _mm512_loadu_si512(xq[i] + size_t(g8 + rr) * 32);
                tile512<M>(win, x, a, K);
            }
            if (pfn && (t & 1) == 0)
                for (int rr = 0; rr < 8; ++rr)
                {
                    const char* q = reinterpret_cast<const char*>(p + size_t(8 * PFG + rr) * row_stride);
                    if (g8 + 8 * PFG >= tk) break;
                    if (K_.pfhint == 0) { _mm_prefetch(q, _MM_HINT_T0); _mm_prefetch(q + 64, _MM_HINT_T0); _mm_prefetch(q + 128, _MM_HINT_T0); }
                    else if (K_.pfhint == 2) { _mm_prefetch(q, _MM_HINT_NTA); _mm_prefetch(q + 64, _MM_HINT_NTA); _mm_prefetch(q + 128, _MM_HINT_NTA); }
                    else { _mm_prefetch(q, _MM_HINT_T1); _mm_prefetch(q + 64, _MM_HINT_T1); _mm_prefetch(q + 128, _MM_HINT_T1); }
                }
            float* af = accf + size_t(t) * 64 * M;
            #pragma GCC unroll 8
            for (int c = 0; c < 8; ++c)
                #pragma GCC unroll 2
                for (int i = 0; i < M; ++i)
                {
                    const __m512i v = a[c * M + i];
                    const __m256i s = _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
                    float* f = af + (c * M + i) * 8;
                    _mm256_store_ps(f, _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc[i], _mm256_load_ps(f)));
                }
        }
    }
    for (int i = 0; i < M; ++i)
        for (int t = 0; t < C; ++t)
            for (int c = 0; c < 8; ++c)
            {
                const float* v = accf + size_t(t) * 64 * M + (c * M + i) * 8;
                y[i * ldy + t * 16 + c] = (v[0] + v[1]) + (v[2] + v[3]);
                y[i * ldy + t * 16 + c + 8] = (v[4] + v[5]) + (v[6] + v[7]);
            }
}

#pragma GCC pop_options
