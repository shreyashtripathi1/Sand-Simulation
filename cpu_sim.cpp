// CPU (OpenMP) port of the 3D sand cube, for benchmarking against the CUDA version.
//
// It is the same algorithm as main.cu, step for step:
//   1. drop sand in a sphere at the dropper
//   2. two simulation steps per frame: each grain tries straight down, then the 8 diagonal
//      cells below, in a hashed, flip-flopped order; conflicts are resolved with a claim
//      flag (compare-and-swap), exactly like the atomicCAS in the CUDA kernel
//   3. visibility pass: keep only grains with an exposed face turned towards the camera,
//      shade them and append them to a compact list (back-face culling for voxels)
// Only the drawing is missing, because there is no GPU to draw with. The CUDA figure to
// compare against is therefore "gpu_ms" (sim + cull), not the wall-clock frame time.
//
// Build (MSVC):  cl /nologo /O2 /openmp /std:c++17 /EHsc cpu_sim.cpp
// Build (g++):   g++ -O2 -fopenmp -std=c++17 cpu_sim.cpp -o cpu_sim
//
// Usage:  cpu_sim [gridSize] [threads=T] [frames=F] [power=battery|ac] [selftest]
//   threads=0 (default) uses every logical core.  Results are appended to
//   benchmark_cpu_results.csv and printed.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int EMPTY = 0;

struct Vertex {
    float x, y, z;
    unsigned char r, g, b, a;
};

static inline int CellIndex(int x, int y, int z, int N) { return (y * N + z) * N + x; }

static const int kDirX[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
static const int kDirZ[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };

static inline unsigned int HashCell(unsigned int x, unsigned int y, unsigned int z, unsigned int step) {
    unsigned int h = (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u) ^ (step * 2654435761u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

// The claim flag: the first thread to swap 0 -> 1 owns the target cell.
static inline bool Claim(std::atomic<unsigned>* claims, int idx) {
    unsigned expected = 0;
    return claims[idx].compare_exchange_strong(expected, 1u, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// 1. Sand drop: fills empty cells inside a sphere. Returns how many grains were added.
// ---------------------------------------------------------------------------
static long AddSand(int* grid, int cx, int cy, int cz, int radius, int N, int simStep) {
    float t = simStep * 0.005f;
    unsigned char r = (unsigned char)((sinf(t)          * 0.5f + 0.5f) * 255.0f);
    unsigned char g = (unsigned char)((sinf(t + 2.094f) * 0.5f + 0.5f) * 255.0f);
    unsigned char b = (unsigned char)((sinf(t + 4.188f) * 0.5f + 0.5f) * 255.0f);
    int packed = (255 << 24) | (r << 16) | (g << 8) | b;

    long added = 0;
    for (int y = cy - radius; y <= cy + radius; y++)
        for (int z = cz - radius; z <= cz + radius; z++)
            for (int x = cx - radius; x <= cx + radius; x++) {
                if (x < 0 || x >= N || y < 0 || y >= N || z < 0 || z >= N) continue;
                int dx = x - cx, dy = y - cy, dz = z - cz;
                if (dx * dx + dy * dy + dz * dz > radius * radius) continue;
                int idx = CellIndex(x, y, z, N);
                if (grid[idx] != EMPTY) continue;
                grid[idx] = packed;
                added++;
            }
    return added;
}

// ---------------------------------------------------------------------------
// 2. One simulation step (ping-pong: reads `in`, writes `out`)
// ---------------------------------------------------------------------------
static void SimulateStep(const int* in, int* out, std::atomic<unsigned>* claims, int N,
                         bool oddFrame, unsigned simStep) {
    // out = in, claims = 0 (same two passes the GPU version does with cudaMemcpy / cudaMemset)
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < N * N; row++) {
        std::memcpy(out + (size_t)row * N, in + (size_t)row * N, (size_t)N * sizeof(int));
        for (int x = 0; x < N; x++) claims[(size_t)row * N + x].store(0u, std::memory_order_relaxed);
    }

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < N * N; row++) {
        int y = row / N;
        int zr = row % N;
        // mirrored thread->cell mapping on alternate steps (see the CUDA kernel)
        int z = (simStep & 2u) ? N - 1 - zr : zr;
        if (y == 0) continue;                              // resting on the floor

        for (int xr = 0; xr < N; xr++) {
            int x = (simStep & 1u) ? N - 1 - xr : xr;
            int idx = CellIndex(x, y, z, N);
            int particle = in[idx];
            if (particle == EMPTY) continue;

            int downIdx = CellIndex(x, y - 1, z, N);
            if (in[downIdx] == EMPTY && Claim(claims, downIdx)) {
                out[idx] = EMPTY;
                out[downIdx] = particle;
                continue;
            }

            unsigned int h = HashCell(x, y, z, simStep);
            int start = (int)(h & 7u);
            int stride = oddFrame ? 1 : 7;
            for (int i = 0; i < 8; i++) {
                int d = (start + i * stride) & 7;
                int nx = x + kDirX[d];
                int nz = z + kDirZ[d];
                if (nx < 0 || nx >= N || nz < 0 || nz >= N) continue;
                int tIdx = CellIndex(nx, y - 1, nz, N);
                if (in[tIdx] == EMPTY && Claim(claims, tIdx)) {
                    out[idx] = EMPTY;
                    out[tIdx] = particle;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Visibility pass: back-face culling + lighting + compaction
// ---------------------------------------------------------------------------
static unsigned BuildVisible(const int* grid, Vertex* out, std::atomic<unsigned>& count,
                             unsigned maxCount, int N, float camX, float camY, float camZ) {
    count.store(0u);
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < N * N; row++) {
        int y = row / N;
        int z = row % N;
        for (int x = 0; x < N; x++) {
            int particle = grid[CellIndex(x, y, z, N)];
            if (particle == EMPTY) continue;

            float vx = camX - (x + 0.5f), vy = camY - (y + 0.5f), vz = camZ - (z + 0.5f);
            float nx = 0.0f, ny = 0.0f, nz = 0.0f;
            bool visible = false;
            for (int f = 0; f < 6; f++) {
                int dx = (f == 0) - (f == 1);
                int dy = (f == 2) - (f == 3);
                int dz = (f == 4) - (f == 5);
                int ax = x + dx, ay = y + dy, az = z + dz;
                bool exposed;
                if (ax < 0 || ax >= N || az < 0 || az >= N || ay < 0) exposed = false;
                else if (ay >= N)                                      exposed = true;
                else exposed = (grid[CellIndex(ax, ay, az, N)] == EMPTY);
                if (!exposed) continue;
                nx += dx; ny += dy; nz += dz;
                if (dx * vx + dy * vy + dz * vz > 0.0f) visible = true;
            }
            if (!visible) continue;

            float len = sqrtf(nx * nx + ny * ny + nz * nz);
            float lambert = 0.0f;
            if (len > 0.0f) lambert = fmaxf(0.0f, (nx * 0.35f + ny * 0.80f + nz * 0.48f) / len);
            float shade = 0.50f + 0.50f * lambert;

            unsigned slot = count.fetch_add(1u, std::memory_order_relaxed);
            if (slot >= maxCount) continue;
            Vertex v;
            v.x = x + 0.5f; v.y = y + 0.5f; v.z = z + 0.5f;
            v.r = (unsigned char)(((particle >> 16) & 0xFF) * shade);
            v.g = (unsigned char)(((particle >> 8)  & 0xFF) * shade);
            v.b = (unsigned char)(( particle        & 0xFF) * shade);
            v.a = 255;
            out[slot] = v;
        }
    }
    unsigned c = count.load();
    return c > maxCount ? maxCount : c;
}

// ---------------------------------------------------------------------------
static std::string CpuName() {
    if (const char* e = getenv("PROCESSOR_IDENTIFIER")) return e;          // Windows
    if (FILE* f = fopen("/proc/cpuinfo", "r")) {                           // Linux
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* c = strchr(line, ':');
                fclose(f);
                if (c) {
                    std::string s = c + 1;
                    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                    size_t i = s.find_first_not_of(' ');
                    return i == std::string::npos ? s : s.substr(i);
                }
                return "unknown CPU";
            }
        }
        fclose(f);
    }
    return "unknown CPU";
}

static double NowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Same 10^3 block / camera as the CUDA culling test: must give exactly the expected count.
static int SelfTest() {
    const int N = 24;
    std::vector<int> g((size_t)N * N * N, 0);
    for (int y = 2; y < 12; y++) for (int z = 6; z < 16; z++) for (int x = 6; x < 16; x++)
        g[CellIndex(x, y, z, N)] = 0xFF8040;
    std::vector<Vertex> out((size_t)N * N * N);
    std::atomic<unsigned> cnt(0);
    float cam[3] = { N / 2 + 200.f, N / 2 + 120.f, N / 2 + 150.f };
    unsigned got = BuildVisible(g.data(), out.data(), cnt, (unsigned)out.size(), N, cam[0], cam[1], cam[2]);
    unsigned expect = 0;
    for (int y = 2; y < 12; y++) for (int z = 6; z < 16; z++) for (int x = 6; x < 16; x++) {
        bool vis = (x == 15 && cam[0] > x) || (x == 6 && cam[0] < x + 1) ||
                   (y == 11 && cam[1] > y) || (y == 2 && cam[1] < y) ||
                   (z == 15 && cam[2] > z) || (z == 6 && cam[2] < z);
        expect += vis;
    }
    printf("selftest culling: emitted %u, expected %u -> %s\n", got, expect, got == expect ? "OK" : "FAIL");
    return got == expect ? 0 : 1;
}

int main(int argc, char** argv) {
    int N = 192, threads = 0, frames = 300;
    std::string power = "unknown";
    bool selftest = false;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "threads=", 8) == 0)      threads = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "frames=", 7) == 0)  frames = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "power=", 6) == 0)   power = argv[i] + 6;
        else if (strcmp(argv[i], "selftest") == 0)     selftest = true;
        else { int n = atoi(argv[i]); if (n >= 16 && n <= 320) N = n; }
    }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    int usedThreads = omp_get_max_threads();
#else
    int usedThreads = 1;
#endif
    if (selftest) return SelfTest();

    const int WARMUP = frames / 6;
    const size_t CELLS = (size_t)N * N * N;
    const unsigned MAX_VERTS = (unsigned)(CELLS < 3000000u ? CELLS : 3000000u);

    std::vector<int> bufA(CELLS, 0), bufB(CELLS, 0);
    int* in = bufA.data();
    int* out = bufB.data();
    std::unique_ptr<std::atomic<unsigned>[]> claims(new std::atomic<unsigned>[CELLS]);
    std::vector<Vertex> verts(MAX_VERTS);
    std::atomic<unsigned> vcount(0);

    const int dropX = N / 2, dropY = N - 4, dropZ = N / 2;
    const int brush = (N / 48 > 1) ? N / 48 : 1;
    const float PI = 3.14159265358979f;
    float yaw = 35.0f;
    const float pitch = 25.0f, dist = N * 2.2f;

    unsigned simStep = 0;
    long totalAdded = 0;
    double sumSim = 0, sumCull = 0, sumVisible = 0;
    int samples = 0;

    for (int f = 0; f < frames; f++) {
        yaw += 0.3f;
        double t0 = NowMs();

        totalAdded += AddSand(in, dropX, dropY, dropZ, brush, N, (int)simStep);
        for (int sub = 0; sub < 2; sub++) {
            SimulateStep(in, out, claims.get(), N, (simStep & 1u) != 0u, simStep);
            int* tmp = in; in = out; out = tmp;
            simStep++;
        }
        double t1 = NowMs();

        float yR = yaw * PI / 180.0f, pR = pitch * PI / 180.0f;
        float camDir[3] = { -cosf(pR) * sinf(yR), sinf(pR), cosf(pR) * cosf(yR) };
        unsigned visible = BuildVisible(in, verts.data(), vcount, MAX_VERTS, N,
                                        N * 0.5f + camDir[0] * dist,
                                        N * 0.5f + camDir[1] * dist,
                                        N * 0.5f + camDir[2] * dist);
        double t2 = NowMs();

        if (f >= WARMUP) {
            sumSim += t1 - t0; sumCull += t2 - t1; sumVisible += visible; samples++;
        }
    }

    long present = 0;
    for (size_t i = 0; i < CELLS; i++) present += (in[i] != EMPTY);
    bool conserved = (present == totalAdded);

    double sim = sumSim / samples, cull = sumCull / samples;
    std::string cpu = CpuName();
    printf("CPU BENCH %s threads=%d grid=%d^3 visible=%.0f sim=%.3f ms cull=%.3f ms total=%.3f ms (%.1f FPS equiv)"
           " power=%s grains=%ld %s\n",
           cpu.c_str(), usedThreads, N, sumVisible / samples, sim, cull, sim + cull,
           1000.0 / (sim + cull), power.c_str(), present, conserved ? "conserved-OK" : "CONSERVATION-FAIL");

    bool exists = false;
    if (FILE* t = fopen("benchmark_cpu_results.csv", "r")) { exists = true; fclose(t); }
    if (FILE* fcsv = fopen("benchmark_cpu_results.csv", "a")) {
        if (!exists)
            fprintf(fcsv, "cpu,threads,power,grid,cells,frames_measured,visible_grains_avg,sim_ms,cull_ms,total_ms,fps_equiv\n");
        fprintf(fcsv, "\"%s\",%d,%s,%d,%zu,%d,%.0f,%.3f,%.3f,%.3f,%.1f\n",
                cpu.c_str(), usedThreads, power.c_str(), N, CELLS, samples, sumVisible / samples,
                sim, cull, sim + cull, 1000.0 / (sim + cull));
        fclose(fcsv);
    }
    return conserved ? 0 : 2;
}
