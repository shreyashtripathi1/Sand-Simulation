// 2D CUDA Falling-Sand Simulation
//
//  Controls:  left mouse = spawn sand,  right mouse = blast / eruption
//  Usage:     sim2d [size=WxH] [block=AxB] [pinned] [novsync] [bench]
//    size=WxH    grid resolution (default 1280x720; the window stays 1280x720)
//    block=AxB   CUDA thread-block shape (default 16x8, at most 1024 threads)
//    pinned      use page-locked host memory for the per-frame colour-buffer copy
//    novsync     turn vsync off so FPS is not capped by the monitor
//    bench       run a fixed scripted scene (sand spawned while the cursor sweeps,
//                periodic blasts), append one row to benchmark_2d_results.csv, exit.
//                Used by run_2d_benchmarks.bat.
//  The window title shows GPU time per frame (simulation / render / copy) and FPS.
//
//  The simulation, blast and render kernels below are unchanged from the original.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

// Global state for mouse
double mouseX = 0, mouseY = 0;
bool isMouseDown = false;
bool isRightMouseDown = false; // NEW

// GLFW callback for mouse clicks
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) isMouseDown = true;
        else if (action == GLFW_RELEASE) isMouseDown = false;
    }
    // Track right clicks
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) isRightMouseDown = true;
        else if (action == GLFW_RELEASE) isRightMouseDown = false;
    }
}

// GLFW callback for mouse movement
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    mouseX = xpos;
    mouseY = ypos;
}

//  Configuration 
const int WIN_W = 1280;   // window size (the grid resolution is chosen at run time)
const int WIN_H = 720;

// States
const int EMPTY = 0;
const int SAND = 1;


// GPU DEVICE CODE 


// 1. The Atomic Traffic Cop
__device__ bool AtomicCheckUnclaimed(unsigned int* claims, int targetIdx) {
    // Attempt to change the claim from 0 to 1. 
    // If it returns 0, we were the first ones here.
    unsigned int previous = atomicCAS(&claims[targetIdx], 0, 1);
    return (previous == 0);
}

// 2. The Compute Shader Kernel
__global__ void SimulateParticlesKernel(int* gridInput, int* gridOutput, unsigned int* claims, int width, int height, bool oddFrame, int mouseX, int mouseY, bool isRightMouseDown, int brushRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int currentIdx = y * width + x;
    int particle = gridInput[currentIdx];

    if (particle == EMPTY) return; 

    // THE ERUPTION BULLDOZER
    if (isRightMouseDown) {
        int gridMouseY = height - 1 - mouseY;
        
        float dx = (float)(x - mouseX);
        float dy = (float)(y - gridMouseY);
        float distSq = dx*dx + dy*dy;
        
        // If the mouse is touching this sand particle...
        if (distSq < (brushRadius * brushRadius) && distSq > 0.1f) {
            float dist = sqrtf(distSq);
            
            // Normalize the direction vector (points directly AWAY from mouse)
            float nx = dx / dist;
            float ny = dy / dist;
            
            // Add a slight upward bias so sand flies into the air
            ny += 0.2f; 

            // Look up to 200 pixels away to find the surface
            int maxBlastDistance = 400; 

            // Search OUTWAR starting from the particle, traveling through solid sand
            for (int step = 1; step < maxBlastDistance; step++) {
                int pushX = x + (int)(nx * step);
                int pushY = y + (int)(ny * step);
                
                if (pushX >= 0 && pushX < width && pushY >= 0 && pushY < height) {
                    int pushIdx = pushY * width + pushX;
                    
                    // The very first EMPTY spot of open air we break through to becomes our new home
                    if (gridInput[pushIdx] == EMPTY && AtomicCheckUnclaimed(claims, pushIdx)) {
                        gridOutput[currentIdx] = EMPTY;   // Leave our buried tomb
                        gridOutput[pushIdx] = particle;   // Pop out on the surface
                        return; // Successfully erupted! Skip normal gravity.
                    }
                } else {
                    break; // We hit the wall or ceiling of the window, stop looking
                }
            }
        }
    }

    // NORMAL GRAVITY
    int downIdx = (y - 1) * width + x;
    int downLeftIdx = (y - 1) * width + (x - 1);
    int downRightIdx = (y - 1) * width + (x + 1);

    int maybeDr = oddFrame ? downRightIdx : downLeftIdx;
    int maybeDl = oddFrame ? downLeftIdx : downRightIdx;

    if (gridInput[downIdx] == EMPTY && AtomicCheckUnclaimed(claims, downIdx)) {
        gridOutput[currentIdx] = EMPTY;   
        gridOutput[downIdx] = particle;       
    }
    else if (gridInput[maybeDr] == EMPTY && AtomicCheckUnclaimed(claims, maybeDr)) {
        gridOutput[currentIdx] = EMPTY;
        gridOutput[maybeDr] = particle;
    }
    else if (gridInput[maybeDl] == EMPTY && AtomicCheckUnclaimed(claims, maybeDl)) {
        gridOutput[currentIdx] = EMPTY;
        gridOutput[maybeDl] = particle;
    }
}

// 3. Render to Color 
__global__ void RenderToColorKernel(int* grid, uchar4* colorBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int particle = grid[idx];

    if (particle != EMPTY) { 
        // Unpack the RGB values using bitwise shifts
        unsigned char r = (particle >> 16) & 0xFF;
        unsigned char g = (particle >> 8) & 0xFF;
        unsigned char b = particle & 0xFF;
        
        colorBuffer[idx] = make_uchar4(r, g, b, 255); 
    } else { 
        colorBuffer[idx] = make_uchar4(30, 30, 30, 255); // Background
    }
}

// 4. Add Sand 
__global__ void AddSandKernel(int* grid, int mouseX, int mouseY, int brushRadius, int width, int height, int simStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int gridMouseY = height - 1 - mouseY; 
    int dx = x - mouseX;
    int dy = y - gridMouseY;
    
    if (dx*dx + dy*dy < brushRadius*brushRadius) {
        
        // Only spawn sand if this specific pixel is empty!
        if (grid[y * width + x] == EMPTY) {
            float waveSpeed = 0.005f;
            float t = simStep * waveSpeed;

            unsigned char r = (unsigned char)((sinf(t) * 0.5f + 0.5f) * 255.0f);
            unsigned char g = (unsigned char)((sinf(t + 2.094f) * 0.5f + 0.5f) * 255.0f); 
            unsigned char b = (unsigned char)((sinf(t + 4.188f) * 0.5f + 0.5f) * 255.0f); 

            int packedColor = (255 << 24) | (r << 16) | (g << 8) | b;
            grid[y * width + x] = packedColor; 
        }
    }
}

// CPU HOST CODE

int main(int argc, char** argv) {
    // ---- options -----------------------------------------------------------
    int W = 1280, H = 720;                 // grid resolution
    int bx = 16, by = 8;                   // thread-block shape
    bool vsync = true, bench = false, pinned = false;
    for (int i = 1; i < argc; i++) {
        int a, b;
        if (strcmp(argv[i], "novsync") == 0)        vsync = false;
        else if (strcmp(argv[i], "pinned") == 0)    pinned = true;
        else if (strcmp(argv[i], "bench") == 0)   { bench = true; vsync = false; }
        else if (sscanf(argv[i], "size=%dx%d", &a, &b) == 2 && a >= 64 && b >= 64 && a <= 7680 && b <= 7680) {
            W = a; H = b;
        } else if (sscanf(argv[i], "block=%dx%d", &a, &b) == 2 && a > 0 && b > 0 && a * b <= 1024) {
            bx = a; by = b;
        } else {
            fprintf(stderr, "Ignoring unknown or invalid option '%s'\n", argv[i]);
        }
    }
    const size_t NUM_PIXELS = (size_t)W * H;
    const int brushSize = (H / 36 < 4) ? 4 : H / 36;      // 20 at 720p, as in the original

    // 1. CUDA Memory Setup
    int *d_gridInput, *d_gridOutput;
    unsigned int *d_claims;
    uchar4 *d_colorBuffer;
    uchar4 *h_colorBuffer;

    CUDA_CHECK(cudaMalloc(&d_gridInput, NUM_PIXELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridOutput, NUM_PIXELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_claims, NUM_PIXELS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_colorBuffer, NUM_PIXELS * sizeof(uchar4)));

    if (pinned) CUDA_CHECK(cudaMallocHost((void**)&h_colorBuffer, NUM_PIXELS * sizeof(uchar4)));
    else        h_colorBuffer = (uchar4*)malloc(NUM_PIXELS * sizeof(uchar4));

    CUDA_CHECK(cudaMemset(d_gridInput, 0, NUM_PIXELS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_gridOutput, 0, NUM_PIXELS * sizeof(int)));

    // 2. OpenGL & GLFW Setup
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "CUDA Sand Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(vsync ? 1 : 0);

    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // Create an OpenGL Texture to display the colors
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);

    // 3. Grid/Block Setup
    dim3 threadsPerBlock(bx, by, 1);
    dim3 numBlocks((W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (H + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    // GPU timing (CUDA events): simulation, render kernel, device-to-host copy
    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    double hudSim = 0, hudRender = 0, hudCopy = 0;
    double bSim = 0, bRender = 0, bCopy = 0, bWall = 0;   // benchmark sums
    const int BENCH_FRAMES = 700, BENCH_WARMUP = 100;
    int benchFrame = 0, benchSamples = 0;
    double titleTimer = 0.0;
    int hudFrames = 0;
    double prevTime = glfwGetTime();

    int simStep = 0;

    // 4. Main Game Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        double now = glfwGetTime();
        double dt = now - prevTime;
        prevTime = now;

        bool oddFrame = (simStep % 2 == 1);

        // Cursor / buttons in grid coordinates
        int mX, mY;
        bool down, right;
        if (bench) {
            // scripted scene: cursor sweeps left-right near the top, sand is always spawned,
            // a blast fires for 25 frames out of every 200
            mX = (int)(W * 0.5f + W * 0.35f * sinf(benchFrame * 0.02f));
            mY = (int)(H * 0.15f);
            down = true;
            right = (benchFrame % 200) >= 150 && (benchFrame % 200) < 175;
        } else {
            int winW, winH;
            glfwGetWindowSize(window, &winW, &winH);
            if (winW < 1) winW = 1;
            if (winH < 1) winH = 1;
            mX = (int)(mouseX * W / winW);
            mY = (int)(mouseY * H / winH);
            down = isMouseDown;
            right = isRightMouseDown;
        }

        CUDA_CHECK(cudaEventRecord(e0));

        // A. INPUT PHASE
        if (down) {
            AddSandKernel<<<numBlocks, threadsPerBlock>>>(d_gridInput, mX, mY, brushSize, W, H, simStep);
        }

        // B. SIMULATION PHASE
        CUDA_CHECK(cudaMemcpy(d_gridOutput, d_gridInput, NUM_PIXELS * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_claims, 0, NUM_PIXELS * sizeof(unsigned int)));
        SimulateParticlesKernel<<<numBlocks, threadsPerBlock>>>(d_gridInput, d_gridOutput, d_claims, W, H, oddFrame, mX, mY, right, brushSize);
        CUDA_CHECK(cudaEventRecord(e1));

        // C. RENDER PHASE
        RenderToColorKernel<<<numBlocks, threadsPerBlock>>>(d_gridOutput, d_colorBuffer, W, H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e2));

        CUDA_CHECK(cudaMemcpy(h_colorBuffer, d_colorBuffer, NUM_PIXELS * sizeof(uchar4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));

        float msSim = 0, msRender = 0, msCopy = 0;
        CUDA_CHECK(cudaEventElapsedTime(&msSim, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&msRender, e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&msCopy, e2, e3));

        // Update texture and draw full screen quad
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_colorBuffer);

        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();

        glfwSwapBuffers(window);

        // D. SWAP BUFFERS (Ping-Pong)
        int* temp = d_gridInput;
        d_gridInput = d_gridOutput;
        d_gridOutput = temp;

        simStep++;

        // ---- benchmark bookkeeping -----------------------------------------
        if (bench) {
            if (benchFrame >= BENCH_WARMUP) {
                bSim += msSim; bRender += msRender; bCopy += msCopy; bWall += dt;
                benchSamples++;
            }
            if (++benchFrame >= BENCH_FRAMES) break;
        }

        // ---- window title as a small HUD -------------------------------------
        hudSim += msSim; hudRender += msRender; hudCopy += msCopy;
        hudFrames++;
        titleTimer += dt;
        if (titleTimer >= 0.5) {
            char title[256];
            snprintf(title, sizeof(title),
                     "CUDA Sand Simulation %dx%d | GPU: sim %.2f + render %.2f + copy %.2f ms | %.0f FPS%s",
                     W, H, hudSim / hudFrames, hudRender / hudFrames, hudCopy / hudFrames,
                     hudFrames / titleTimer, vsync ? " (vsync)" : " (no vsync)");
            glfwSetWindowTitle(window, title);
            titleTimer = 0.0; hudFrames = 0; hudSim = hudRender = hudCopy = 0.0;
        }
    }

    // Benchmark result: one CSV row per run
    if (bench && benchSamples > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        double sim = bSim / benchSamples, render = bRender / benchSamples, copy = bCopy / benchSamples;
        double wall = bWall / benchSamples * 1000.0;
        bool exists = false;
        if (FILE* t = fopen("benchmark_2d_results.csv", "r")) { exists = true; fclose(t); }
        if (FILE* f = fopen("benchmark_2d_results.csv", "a")) {
            if (!exists)
                fprintf(f, "gpu,width,height,block,pinned,cells,frames_measured,sim_ms,render_ms,copy_ms,gpu_ms,frame_ms,fps\n");
            fprintf(f, "\"%s\",%d,%d,%dx%d,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
                    prop.name, W, H, bx, by, pinned ? 1 : 0, NUM_PIXELS, benchSamples,
                    sim, render, copy, sim + render + copy, wall, 1000.0 / wall);
            fclose(f);
        }
        printf("BENCH2D %s %dx%d block=%dx%d pinned=%d sim=%.3f render=%.3f copy=%.3f ms frame=%.3f ms (%.1f FPS)\n",
               prop.name, W, H, bx, by, pinned ? 1 : 0, sim, render, copy, wall, 1000.0 / wall);
    }

    // 5. Cleanup
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaFree(d_gridInput);
    cudaFree(d_gridOutput);
    cudaFree(d_claims);
    cudaFree(d_colorBuffer);
    if (pinned) cudaFreeHost(h_colorBuffer); else free(h_colorBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
