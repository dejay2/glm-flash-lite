// AVX-512 vs AVX2 p16 kernel: randomized bit-for-bit comparison over real experts of the EXL3 expert store (read-only), plus
// the ft_core fp64-class error of both. For every sampled expert: 1 token and 2 tokens routed to it, alone (n=1) and in
// groups of 6 experts, default band shapes and ft_core-shaped block units, auto mode (I16) and explicit AFFINE/EXACT
// (those never use the AVX-512 kernel, so they must match trivially).
// usage: isa_compare STORE.bin [nbatch=4] [per=96] [threads=13] [cpu0=2] [seed=1] [cpustep=2]
#include "ft_n135.h"
#include <map>
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: isa_compare STORE.bin [k=v ...]\n"); return 2; }
    map<string, string> A;
    for (int i = 2; i < argc; ++i) { string s = argv[i]; auto p = s.find('='); if (p != string::npos) A[s.substr(0, p)] = s.substr(p + 1); }
    auto arg = [&](const char* k, int d) { auto it = A.find(k); return it == A.end() ? d : stoi(it->second); };
    const int NB = arg("nbatch", 4), PER = arg("per", 96), T = arg("threads", 13), CPU0 = arg("cpu0", 2), SEED = arg("seed", 1), STEP = arg("cpustep", 2);
    if (!n135::cpu_has_512()) { fprintf(stderr, "CPU lacks AVX-512 f/bw/vl/dq/vbmi/vnni\n"); return 3; }
    g_act_limit = 10.f;
    init_perm(); init_tables(); n135::init_p16();
    const size_t REC = 9474048, SLOT = 9437184, OFF_U = 3145728, OFF_D = 6291456, NREC_STORE = 43 * 288;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open store"); return 1; }
    vector<int> cpus; for (int i = 0; i < T; ++i) cpus.push_back(CPU0 + STEP * i);
    Pool pool; pool.start(T, cpus);
    mt19937_64 rng(SEED);
    vector<float> x(size_t(2) * 4096);
    { mt19937 r(7); normal_distribution<float> nd(0.f, 1.f); for (auto& v : x) v = r16(nd(r) * 0.5f); }
    size_t tot = 0, bad = 0, nonid_core = 0; double worst_core = 0, sum_core = 0; size_t ncore = 0;
    for (int b = 0; b < NB; ++b)
    {
        vector<size_t> ids;
        while (int(ids.size()) < PER) { size_t r = rng() % NREC_STORE; if (find(ids.begin(), ids.end(), r) == ids.end()) ids.push_back(r); }
        uint8_t* ram = static_cast<uint8_t*>(big_alloc(SLOT * PER));
        vector<uint16_t> scales(size_t(PER) * (REC - SLOT) / 2);
        vector<uint8_t> buf(REC);
        for (int r = 0; r < PER; ++r)
        {
            size_t got = 0;
            while (got < REC) { ssize_t k = pread(fd, buf.data() + got, REC - got, off_t(ids[r] * REC + got)); if (k <= 0) { perror("pread"); return 1; } got += k; }
            memcpy(ram + size_t(r) * SLOT, buf.data(), SLOT);
            memcpy(scales.data() + size_t(r) * (REC - SLOT) / 2, buf.data() + SLOT, REC - SLOT);
        }
        Layer L; L.H = 4096; L.I = 2048; L.ex.resize(PER);
        for (int r = 0; r < PER; ++r)
        {
            const uint16_t* sc = scales.data() + size_t(r) * (REC - SLOT) / 2;
            uint8_t* a = ram + size_t(r) * SLOT;
            L.ex[r].g = { a, sc, sc + 4096, 4096, 2048 };
            L.ex[r].u = { a + OFF_U, sc + 6144, sc + 6144 + 4096, 4096, 2048 };
            L.ex[r].d = { a + OFF_D, sc + 12288, sc + 12288 + 2048, 2048, 4096 };
        }
        vector<float> o1(size_t(2) * 4096), o2(size_t(2) * 4096), oc(size_t(2) * 4096);
        for (int bandmaxm : { 1, 8 })
        {
            n135::K_.bandmaxm = bandmaxm;
            for (int m : { 1, 2 })
                for (int grp : { 1, 6 })
                    for (int s = 0; s + grp <= PER; s += grp)
                    {
                        vector<vector<pair<int, float>>> route(m);
                        for (int t = 0; t < m; ++t) for (int i = 0; i < grp; ++i) route[t].push_back({ s + i, 0.1f + 0.05f * i });
                        n135::select_isa("avx2");
                        n135::moe_forward4(pool, L, x.data(), m, route, o1.data(), -1);
                        n135::select_isa("avx512");
                        if (!n135::use512()) { fprintf(stderr, "avx512 not active\n"); return 4; }
                        n135::moe_forward4(pool, L, x.data(), m, route, o2.data(), -1);
                        tot += size_t(m) * 4096;
                        size_t nd = 0; for (size_t i = 0; i < size_t(m) * 4096; ++i) nd += memcmp(&o1[i], &o2[i], 4) != 0;
                        bad += nd;
                        if (nd) fprintf(stderr, "MISMATCH batch %d bandmaxm %d m %d grp %d start %d: %zu values differ\n", b, bandmaxm, m, grp, s, nd);
                        if (bandmaxm == 1 && grp == 6)
                        {
                            moe_forward(pool, L, x.data(), m, route, oc.data(), 2);   // ft_core I16 reference
                            double num = 0, den = 0; for (size_t i = 0; i < size_t(m) * 4096; ++i) { double d = double(o2[i]) - oc[i]; num += d * d; den += double(oc[i]) * oc[i]; }
                            const double rel = sqrt(num / den); worst_core = max(worst_core, rel); sum_core += rel; ++ncore;
                        }
                    }
        }
        printf("batch %d/%d done: %zu values compared, %zu differ so far\n", b + 1, NB, tot, bad); fflush(stdout);
        free(ram);
    }
    printf("RESULT avx512 vs avx2 (n135 p16, auto/I16): %zu values, %zu bitwise differences; avx512 vs ft_core I16 rel_rms mean %.3e worst %.3e (%zu calls)\n",
           tot, bad, sum_core / max<size_t>(ncore, 1), worst_core, ncore);
    pool.stop();
    return bad ? 1 : 0;
}
