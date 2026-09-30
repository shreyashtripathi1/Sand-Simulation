// 3D CUDA Falling-Sand Cube
//
//  - Thread-per-cell CUDA kernel over an N x N x N voxel grid (ping-pong buffers)
//  - Flip-flop neighbour ordering + atomicCAS claims => race-free particle moves
//  - Rendered as a transparent glass cube (OpenGL fixed-function pipeline)
//
//  Controls
//    Left mouse (hold + drag) : rotate the cube          Mouse wheel : zoom
//    Arrow keys               : move dropper horizontally (relative to the view)
//    Space / Ctrl             : move dropper up / down
//    Enter (hold)             : drop sand
//    [ / ]                    : dropper size             C : clear     R : reset camera
//    Esc                      : quit
//
//  Usage:  sim3d [gridSize] [novsync]      (default 192, allowed 16..320)
//          novsync turns vsync off so the FPS is no longer capped by the monitor.
//          block=AxBxC sets the CUDA thread-block shape (default 32x4x2, max 1024 threads).
//          bench runs a fixed scripted scene (sand dropped continuously, camera slowly
//          orbiting, vsync off), appends one row to benchmark_results.csv and exits.
//          Used by run_benchmarks.bat.
//  The window title shows GPU time per frame for the simulation and the culling pass
//  (measured with CUDA events), which is the real cost independent of vsync.
//
//  Rendering: one GPU thread per cell decides whether its grain is visible
//  (has an exposed face pointing at the camera = back-face culling for voxels),
//  shades it, and appends it to a compact vertex list. Only that list is copied
//  to the CPU, so grains can be roughly one pixel in size.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t e_ = (call);                                                    \
        if (e_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_),   \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

const int EMPTY = 0;

struct Vertex {
    float x, y, z;
    unsigned char r, g, b, a;
};

// ==== KERNELS BEGIN (extracted verbatim by the CPU test harness) ====

// Index layout: x fastest, then z, then y (y is "up"). Neighbouring threads in x
// touch neighbouring words -> coalesced reads.
__host__ __device__ inline int CellIndex(int x, int y, int z, int N) {
    return (y * N + z) * N + x;
}

// The 8 horizontal neighbours, listed as a ring so that "rotating" the start
// index / direction gives an unbiased choice.
__constant__ int c_dirX[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
__constant__ int c_dirZ[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };

__device__ inline unsigned int HashCell(unsigned int x, unsigned int y, unsigned int z, unsigned int step) {
    unsigned int h = (x * 73856093u) ^ (y * 19349663u) ^ (z * 83492791u) ^ (step * 2654435761u);
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

// The atomic traffic cop: the first thread to claim a target cell wins.
__device__ inline bool AtomicCheckUnclaimed(unsigned int* claims, int targetIdx) {
    unsigned int previous = atomicCAS(&claims[targetIdx], 0u, 1u);
    return (previous == 0u);
}

// One thread per cell. Sand tries to fall straight down; if blocked it slides
// diagonally down into one of the 8 surrounding columns. The order in which the
// 8 diagonals are tried is randomised per cell (hash) and its rotation direction
// flips every step (flip-flop), which removes directional bias.
__global__ void SimulateParticlesKernel(const int* gridInput, int* gridOutput,
                                        unsigned int* claims, int N,
                                        bool oddFrame, unsigned int simStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= N) return;

    // Flip-flop the thread->cell mapping. When two grains want the same empty cell,
    // the atomic winner tends to be the lower thread index; mirroring the mapping on
    // alternate steps reverses that priority so the pile does not drift to one side.
    if (simStep & 1u) x = N - 1 - x;
    if (simStep & 2u) z = N - 1 - z;

    int idx = CellIndex(x, y, z, N);
    int particle = gridInput[idx];
    if (particle == EMPTY) return;
    if (y == 0) return;                       // resting on the floor

    // 1. straight down
    int downIdx = CellIndex(x, y - 1, z, N);
    if (gridInput[downIdx] == EMPTY && AtomicCheckUnclaimed(claims, downIdx)) {
        gridOutput[idx] = EMPTY;
        gridOutput[downIdx] = particle;
        return;
    }

    // 2. diagonally down (8 directions, flip-flop order)
    unsigned int h = HashCell(x, y, z, simStep);
    int start = (int)(h & 7u);
    int stride = oddFrame ? 1 : 7;            // +1 or -1 (mod 8)
    for (int i = 0; i < 8; i++) {
        int d = (start + i * stride) & 7;
        int nx = x + c_dirX[d];
        int nz = z + c_dirZ[d];
        if (nx < 0 || nx >= N || nz < 0 || nz >= N) continue;   // cube walls

        int tIdx = CellIndex(nx, y - 1, nz, N);
        if (gridInput[tIdx] == EMPTY && AtomicCheckUnclaimed(claims, tIdx)) {
            gridOutput[idx] = EMPTY;
            gridOutput[tIdx] = particle;
            return;
        }
    }
}

// Fills a small sphere of empty cells around the dropper with (rainbow) sand.
__global__ void AddSandKernel(int* grid, int cx, int cy, int cz, int radius,
                              int N, int simStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= N) return;

    int dx = x - cx, dy = y - cy, dz = z - cz;
    if (dx * dx + dy * dy + dz * dz > radius * radius) return;

    int idx = CellIndex(x, y, z, N);
    if (grid[idx] != EMPTY) return;           // only spawn into empty cells

    float t = simStep * 0.005f;
    unsigned char r = (unsigned char)((sinf(t)          * 0.5f + 0.5f) * 255.0f);
    unsigned char g = (unsigned char)((sinf(t + 2.094f) * 0.5f + 0.5f) * 255.0f);
    unsigned char b = (unsigned char)((sinf(t + 4.188f) * 0.5f + 0.5f) * 255.0f);
    grid[idx] = (255 << 24) | (r << 16) | (g << 8) | b;
}

// Back-face culling for voxels + lighting + stream compaction.
// A grain is emitted only if at least one of its 6 faces is (a) exposed to air and
// (b) turned towards the camera. Fully buried grains and grains whose exposed faces
// all point away from the viewer can never be seen, so they are skipped.
__global__ void BuildVisibleVoxelsKernel(const int* grid, Vertex* out, unsigned int* count,
                                         unsigned int maxCount, int N,
                                         float camX, float camY, float camZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= N) return;

    int particle = grid[CellIndex(x, y, z, N)];
    if (particle == EMPTY) return;

    // vector from this grain to the camera
    float vx = camX - (x + 0.5f), vy = camY - (y + 0.5f), vz = camZ - (z + 0.5f);

    float nx = 0.0f, ny = 0.0f, nz = 0.0f;    // sum of exposed face normals (for shading)
    bool visible = false;
    for (int f = 0; f < 6; f++) {
        int dx = (f == 0) - (f == 1);
        int dy = (f == 2) - (f == 3);
        int dz = (f == 4) - (f == 5);
        int ax = x + dx, ay = y + dy, az = z + dz;
        // the cube walls are not open air; the open top (y == N) is
        bool exposed;
        if (ax < 0 || ax >= N || az < 0 || az >= N || ay < 0) exposed = false;
        else if (ay >= N)                          exposed = true;
        else                                       exposed = (grid[CellIndex(ax, ay, az, N)] == EMPTY);
        if (!exposed) continue;
        nx += dx; ny += dy; nz += dz;
        if (dx * vx + dy * vy + dz * vz > 0.0f) visible = true;   // faces the camera
    }
    if (!visible) return;

    // simple directional light on the averaged surface normal
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    float lambert = 0.0f;
    if (len > 0.0f) lambert = fmaxf(0.0f, (nx * 0.35f + ny * 0.80f + nz * 0.48f) / len);
    float shade = 0.50f + 0.50f * lambert;

    unsigned int slot = atomicAdd(count, 1u);
    if (slot >= maxCount) return;
    Vertex v;
    v.x = x + 0.5f; v.y = y + 0.5f; v.z = z + 0.5f;
    v.r = (unsigned char)(((particle >> 16) & 0xFF) * shade);
    v.g = (unsigned char)(((particle >> 8)  & 0xFF) * shade);
    v.b = (unsigned char)(( particle        & 0xFF) * shade);
    v.a = 255;
    out[slot] = v;
}

// ==== KERNELS END ====


// ---------------------------------------------------------------------------
// Host side: state
// ---------------------------------------------------------------------------
static const int   WIN_W = 1280;
static const int   WIN_H = 720;
static const float PI_F  = 3.14159265358979f;
static const float FOV_Y = 45.0f;             // degrees

static int   g_N = 192;

// camera (orbits the cube centre)
static float g_yaw   = 35.0f;
static float g_pitch = 25.0f;
static float g_dist  = 140.0f;
static bool  g_dragging = false;
static double g_lastX = 0, g_lastY = 0;

// dropper
static int  g_dropX = 32, g_dropY = 60, g_dropZ = 32;
static int  g_brush = 4;
static bool g_clearRequested = false;
static const int SIM_STEPS_PER_FRAME = 2;    // grains fall this many cells per rendered frame

static void ResetCamera() {
    g_yaw = 35.0f;
    g_pitch = 25.0f;
    g_dist = g_N * 2.2f;
}

// ---------------------------------------------------------------------------
// GLFW callbacks
// ---------------------------------------------------------------------------
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_dragging = true;
            glfwGetCursorPos(window, &g_lastX, &g_lastY);
        } else if (action == GLFW_RELEASE) {
            g_dragging = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow*, double x, double y) {
    if (g_dragging) {
        g_yaw   += (float)(x - g_lastX) * 0.4f;
        g_pitch += (float)(y - g_lastY) * 0.4f;
        if (g_pitch >  89.0f) g_pitch =  89.0f;
        if (g_pitch < -89.0f) g_pitch = -89.0f;
    }
    g_lastX = x;
    g_lastY = y;
}

static void scrollCallback(GLFWwindow*, double, double yoff) {
    g_dist *= (float)pow(0.92, yoff);
    float lo = g_N * 0.8f, hi = g_N * 6.0f;
    if (g_dist < lo) g_dist = lo;
    if (g_dist > hi) g_dist = hi;
}

static void keyCallback(GLFWwindow* window, int key, int, int action, int mods) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(window, GLFW_TRUE); return; }
    if (mods & GLFW_MOD_CONTROL) return;      // Ctrl is the "move down" key
    switch (key) {
        case GLFW_KEY_C:             g_clearRequested = true; break;
        case GLFW_KEY_R:             ResetCamera(); break;
        case GLFW_KEY_LEFT_BRACKET:  if (g_brush > 0) g_brush--; break;
        case GLFW_KEY_RIGHT_BRACKET: if (g_brush < g_N / 8) g_brush++; break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Drawing helpers (fixed-function OpenGL)
// ---------------------------------------------------------------------------
static void DrawWireBox(float x0, float y0, float z0, float x1, float y1, float z1) {
    const float c[8][3] = {
        {x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1},
        {x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1}
    };
    static const int e[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
    };
    glBegin(GL_LINES);
    for (int i = 0; i < 12; i++) {
        glVertex3fv(c[e[i][0]]);
        glVertex3fv(c[e[i][1]]);
    }
    glEnd();
}

// Draws either the faces that point towards the camera (front) or away (back).
// Back faces go behind the sand, front faces in front of it, so the sand looks
// like it is inside a pane of glass.
static void DrawCubeFaces(float S, const float camDir[3], bool front, float alpha) {
    static const float nrm[6][3] = {
        { 1,0,0},{-1,0,0},{0, 1,0},{0,-1,0},{0,0, 1},{0,0,-1}
    };
    static const float cor[6][4][3] = {
        {{1,0,0},{1,1,0},{1,1,1},{1,0,1}},   // +x
        {{0,0,0},{0,0,1},{0,1,1},{0,1,0}},   // -x
        {{0,1,0},{0,1,1},{1,1,1},{1,1,0}},   // +y
        {{0,0,0},{1,0,0},{1,0,1},{0,0,1}},   // -y (floor)
        {{0,0,1},{1,0,1},{1,1,1},{0,1,1}},   // +z
        {{0,0,0},{0,1,0},{1,1,0},{1,0,0}}    // -z
    };
    glBegin(GL_QUADS);
    for (int f = 0; f < 6; f++) {
        float d = nrm[f][0]*camDir[0] + nrm[f][1]*camDir[1] + nrm[f][2]*camDir[2];
        if ((d > 0.0f) != front) continue;
        float a = (f == 3) ? alpha * 2.5f : alpha;          // floor a bit stronger
        glColor4f(0.55f, 0.72f, 1.0f, a);
        for (int v = 0; v < 4; v++)
            glVertex3f(cor[f][v][0]*S, cor[f][v][1]*S, cor[f][v][2]*S);
    }
    glEnd();
}

static void DrawFloorGrid(float S, int divisions) {
    glColor4f(0.6f, 0.75f, 1.0f, 0.18f);
    glBegin(GL_LINES);
    for (int i = 0; i <= divisions; i++) {
        float t = S * i / divisions;
        glVertex3f(t, 0.0f, 0.0f); glVertex3f(t, 0.0f, S);
        glVertex3f(0.0f, 0.0f, t); glVertex3f(S, 0.0f, t);
    }
    glEnd();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool vsync = true;
    bool bench = false;
    int bx = 32, by = 4, bz = 2;              // thread-block shape (x is the memory-contiguous axis)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "novsync") == 0 || strcmp(argv[i], "-novsync") == 0 ||
            strcmp(argv[i], "--novsync") == 0) {
            vsync = false;
        } else if (strncmp(argv[i], "block=", 6) == 0) {
            int a, b, c;
            if (sscanf(argv[i] + 6, "%dx%dx%d", &a, &b, &c) == 3 && a > 0 && b > 0 && c > 0 &&
                a * b * c <= 1024) {
                bx = a; by = b; bz = c;
            } else {
                fprintf(stderr, "Ignoring invalid %s (use block=AxBxC with at most 1024 threads)\n", argv[i]);
            }
        } else if (strcmp(argv[i], "bench") == 0) {
            bench = true;
            vsync = false;
        } else {
            int n = atoi(argv[i]);
            if (n > 0) {
                if (n < 16) n = 16;
                if (n > 320) n = 320;
                g_N = n;
            }
        }
    }
    const int N = g_N;
    const size_t NUM_CELLS = (size_t)N * N * N;

    g_dropX = N / 2;
    g_dropZ = N / 2;
    g_dropY = N - 4;
    g_brush = (N / 48 > 1) ? N / 48 : 1;     // dropper size scales with the grid
    ResetCamera();

    // 1. CUDA memory
    int *d_gridInput = nullptr, *d_gridOutput = nullptr;
    unsigned int* d_claims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gridInput,  NUM_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridOutput, NUM_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_claims,     NUM_CELLS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_gridInput,  0, NUM_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_gridOutput, 0, NUM_CELLS * sizeof(int)));

    // GPU-side list of visible grains (filled by BuildVisibleVoxelsKernel) + host copy for drawing
    const unsigned int MAX_VERTS = (unsigned int)((NUM_CELLS < 3000000u) ? NUM_CELLS : 3000000u);
    Vertex* d_verts = nullptr;
    unsigned int* d_vertCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_verts, (size_t)MAX_VERTS * sizeof(Vertex)));
    CUDA_CHECK(cudaMalloc(&d_vertCount, sizeof(unsigned int)));
    std::vector<Vertex> h_verts(MAX_VERTS);
    unsigned int visibleCount = 0;

    // 2. OpenGL / GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "CUDA Sand Cube 3D", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(vsync ? 1 : 0);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    // 3. Launch configuration: x is the fastest axis (coalesced), blocks of 32x4x2 = 256 threads
    dim3 threadsPerBlock(bx, by, bz);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    unsigned int simStep = 0;
    double prevTime = glfwGetTime();
    double moveAccum = 0.0;
    const double MOVE_TICK = 2.0 / N;         // dropper speed: N/2 cells per second while a key is held
    double titleTimer = 0.0;
    int framesSinceTitle = 0;

    // GPU timing (CUDA events): simulation steps and the visibility/culling pass
    cudaEvent_t evStart, evSim, evCull;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evSim));
    CUDA_CHECK(cudaEventCreate(&evCull));
    double simMsSum = 0.0, cullMsSum = 0.0;

    // benchmark bookkeeping (only used with the "bench" argument)
    const int BENCH_FRAMES = 700, BENCH_WARMUP = 100;
    int benchFrame = 0, benchSamples = 0;
    double bSim = 0, bCull = 0, bWall = 0, bVisible = 0;

    // 4. Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        double rawDt = now - prevTime;
        double dt = rawDt;
        prevTime = now;
        if (dt > 0.1) dt = 0.1;

        // ---- Dropper movement (keyboard) ----------------------------------
        // Arrow keys are relative to the current view, snapped to the nearest
        // grid axis: "up" always moves the dropper away from you, "right" to
        // your right, no matter how you have rotated the cube.
        float yawR = g_yaw * PI_F / 180.0f;
        float fx = sinf(yawR), fz = -cosf(yawR);           // forward on the ground plane
        int fdx = 0, fdz = 0;
        if (fabsf(fx) > fabsf(fz)) fdx = (fx > 0) ? 1 : -1; else fdz = (fz > 0) ? 1 : -1;
        int rdx = -fdz, rdz = fdx;                          // right = forward x up

        bool kFwd   = glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS;
        bool kBack  = glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS;
        bool kLeft  = glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS;
        bool kRight = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
        bool kUp    = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        bool kDown  = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (kFwd || kBack || kLeft || kRight || kUp || kDown) {
            moveAccum += dt;
            while (moveAccum >= MOVE_TICK) {
                moveAccum -= MOVE_TICK;
                if (kFwd)   { g_dropX += fdx; g_dropZ += fdz; }
                if (kBack)  { g_dropX -= fdx; g_dropZ -= fdz; }
                if (kRight) { g_dropX += rdx; g_dropZ += rdz; }
                if (kLeft)  { g_dropX -= rdx; g_dropZ -= rdz; }
                if (kUp)    g_dropY++;
                if (kDown)  g_dropY--;
            }
        } else {
            moveAccum = MOVE_TICK;            // first press moves immediately
        }
        if (g_dropX < 0)     g_dropX = 0;
        if (g_dropX > N - 1) g_dropX = N - 1;
        if (g_dropY < 0)     g_dropY = 0;
        if (g_dropY > N - 1) g_dropY = N - 1;
        if (g_dropZ < 0)     g_dropZ = 0;
        if (g_dropZ > N - 1) g_dropZ = N - 1;

        // ---- Clear ---------------------------------------------------------
        if (g_clearRequested) {
            g_clearRequested = false;
            CUDA_CHECK(cudaMemset(d_gridInput,  0, NUM_CELLS * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_gridOutput, 0, NUM_CELLS * sizeof(int)));
        }

        // ---- A. Input phase: drop sand at the dropper while Enter is held --
        bool dropping = glfwGetKey(window, GLFW_KEY_ENTER)    == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
        CUDA_CHECK(cudaEventRecord(evStart));
        if (bench) {                              // scripted scene: constant dropping + slow orbit
            dropping = true;
            g_yaw += 0.3f;
        }
        if (dropping) {
            AddSandKernel<<<numBlocks, threadsPerBlock>>>(d_gridInput, g_dropX, g_dropY, g_dropZ,
                                                          g_brush, N, (int)simStep);
        }

        // ---- B. Simulation phase (ping-pong + atomic claims) --------------
        for (int sub = 0; sub < SIM_STEPS_PER_FRAME; sub++) {
            bool oddFrame = (simStep & 1u) != 0u;
            CUDA_CHECK(cudaMemcpy(d_gridOutput, d_gridInput, NUM_CELLS * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemset(d_claims, 0, NUM_CELLS * sizeof(unsigned int)));
            SimulateParticlesKernel<<<numBlocks, threadsPerBlock>>>(d_gridInput, d_gridOutput, d_claims,
                                                                    N, oddFrame, simStep);
            CUDA_CHECK(cudaGetLastError());
            // ping-pong: d_gridInput is always the newest state
            int* tmp = d_gridInput; d_gridInput = d_gridOutput; d_gridOutput = tmp;
            simStep++;
        }

        CUDA_CHECK(cudaEventRecord(evSim));

        // ---- C. Camera (needed for culling, glass faces and drawing) ------
        float pR = g_pitch * PI_F / 180.0f;
        float camDir[3] = { -cosf(pR) * sinf(yawR), sinf(pR), cosf(pR) * cosf(yawR) };
        float camPos[3] = { N * 0.5f + camDir[0] * g_dist,
                            N * 0.5f + camDir[1] * g_dist,
                            N * 0.5f + camDir[2] * g_dist };

        // ---- D. GPU builds the list of visible grains (back-face culled) --
        CUDA_CHECK(cudaMemset(d_vertCount, 0, sizeof(unsigned int)));
        BuildVisibleVoxelsKernel<<<numBlocks, threadsPerBlock>>>(d_gridInput, d_verts, d_vertCount,
                                                                 MAX_VERTS, N,
                                                                 camPos[0], camPos[1], camPos[2]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evCull));
        CUDA_CHECK(cudaMemcpy(&visibleCount, d_vertCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        {   // the blocking memcpy above guarantees all three events have completed
            float msSim = 0.0f, msCull = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&msSim,  evStart, evSim));
            CUDA_CHECK(cudaEventElapsedTime(&msCull, evSim,   evCull));
            simMsSum  += msSim;
            cullMsSum += msCull;
            if (bench && benchFrame >= BENCH_WARMUP) {
                bSim += msSim; bCull += msCull; bWall += rawDt; bVisible += visibleCount;
                benchSamples++;
            }
        }
        if (visibleCount > MAX_VERTS) visibleCount = MAX_VERTS;
        if (visibleCount > 0)
            CUDA_CHECK(cudaMemcpy(h_verts.data(), d_verts, (size_t)visibleCount * sizeof(Vertex),
                                  cudaMemcpyDeviceToHost));

        // ---- E. Render -----------------------------------------------------
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        if (fbh < 1) fbh = 1;
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (float)fbw / (float)fbh;
        float nearP = 1.0f, farP = g_dist + 4.0f * N;
        float top = nearP * tanf(FOV_Y * 0.5f * PI_F / 180.0f);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(-top * aspect, top * aspect, -top, top, nearP, farP);

        float c = N * 0.5f;
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -g_dist);
        glRotatef(g_pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(g_yaw,   0.0f, 1.0f, 0.0f);
        glTranslatef(-c, -c, -c);

        glDisable(GL_TEXTURE_2D);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // glass: floor grid + far faces first (behind everything, no depth needed)
        glDisable(GL_DEPTH_TEST);
        DrawFloorGrid((float)N, 8);
        DrawCubeFaces((float)N, camDir, false, 0.05f);

        // sand: opaque, depth-tested points (the grains are already culled on the GPU)
        if (visibleCount > 0) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            // one cell on screen, in pixels; rounded UP so neighbouring grains never leave holes
            float pxPerUnit = fbh / (2.0f * tanf(FOV_Y * 0.5f * PI_F / 180.0f) * g_dist);
            float ps = ceilf(pxPerUnit);
            if (ps < 1.0f) ps = 1.0f;
            glPointSize(ps);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);
            glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &h_verts[0].x);
            glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Vertex), &h_verts[0].r);
            glDrawArrays(GL_POINTS, 0, (GLsizei)visibleCount);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
        }

        // glass: near faces (blended over the sand; depth-test on, depth-write off) + edges
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        DrawCubeFaces((float)N, camDir, true, 0.05f);
        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(1.5f);
        glColor4f(0.7f, 0.85f, 1.0f, 0.75f);
        DrawWireBox(0, 0, 0, (float)N, (float)N, (float)N);

        // dropper marker: brush box + guide line and footprint on the floor
        {
            float r = (float)g_brush;
            float bx0 = g_dropX - r,     bx1 = g_dropX + r + 1.0f;
            float by0 = g_dropY - r,     by1 = g_dropY + r + 1.0f;
            float bz0 = g_dropZ - r,     bz1 = g_dropZ + r + 1.0f;
            if (dropping) glColor4f(1.0f, 0.85f, 0.2f, 0.95f);
            else        glColor4f(1.0f, 0.3f, 0.3f, 0.95f);
            glLineWidth(2.0f);
            DrawWireBox(bx0, by0, bz0, bx1, by1, bz1);
            glLineWidth(1.0f);
            float mx = g_dropX + 0.5f, mz = g_dropZ + 0.5f;
            glBegin(GL_LINES);
                glVertex3f(mx, by0, mz);
                glVertex3f(mx, 0.0f, mz);
            glEnd();
            DrawWireBox(bx0, 0.0f, bz0, bx1, 0.0f, bz1);       // footprint on the floor
        }
        glDisable(GL_BLEND);

        glfwSwapBuffers(window);

        if (bench && ++benchFrame >= BENCH_FRAMES) break;

        // ---- Window title as a tiny HUD ------------------------------------
        framesSinceTitle++;
        titleTimer += dt;
        if (titleTimer >= 0.5) {
            char title[320];
            snprintf(title, sizeof(title),
                     "CUDA Sand Cube %d^3 | dropper (%d, %d, %d) r=%d | %u visible | GPU: sim %.2f ms + cull %.2f ms | %.0f FPS%s",
                     N, g_dropX, g_dropY, g_dropZ, g_brush, visibleCount,
                     simMsSum / framesSinceTitle, cullMsSum / framesSinceTitle,
                     framesSinceTitle / titleTimer, vsync ? " (vsync)" : " (no vsync)");
            glfwSetWindowTitle(window, title);
            titleTimer = 0.0;
            framesSinceTitle = 0;
            simMsSum = cullMsSum = 0.0;
        }
    }

    // Benchmark result: one CSV row per run
    if (bench && benchSamples > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        double avgSim = bSim / benchSamples, avgCull = bCull / benchSamples;
        double avgWall = bWall / benchSamples * 1000.0;     // ms per frame (wall clock)
        bool fileExists = false;
        if (FILE* t = fopen("benchmark_results.csv", "r")) { fileExists = true; fclose(t); }
        if (FILE* f = fopen("benchmark_results.csv", "a")) {
            if (!fileExists)
                fprintf(f, "gpu,grid,block,cells,frames_measured,visible_grains_avg,sim_ms,cull_ms,gpu_ms,frame_ms,fps\n");
            fprintf(f, "\"%s\",%d,%dx%dx%d,%zu,%d,%.0f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
                    prop.name, N, bx, by, bz, NUM_CELLS, benchSamples, bVisible / benchSamples,
                    avgSim, avgCull, avgSim + avgCull, avgWall, 1000.0 / avgWall);
            fclose(f);
        }
        printf("BENCH %s grid=%d^3 block=%dx%dx%d visible=%.0f sim=%.3f ms cull=%.3f ms frame=%.3f ms (%.1f FPS)\n",
               prop.name, N, bx, by, bz, bVisible / benchSamples, avgSim, avgCull, avgWall, 1000.0 / avgWall);
    }

    // 5. Cleanup
    cudaFree(d_gridInput);
    cudaFree(d_gridOutput);
    cudaFree(d_claims);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evSim);
    cudaEventDestroy(evCull);
    cudaFree(d_verts);
    cudaFree(d_vertCount);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
