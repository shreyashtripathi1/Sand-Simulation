#include <fstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h> // You will need to install and link GLFW

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

// Configuration
const int WIDTH = 800;  //This is test size ,Real size 800
const int HEIGHT = 600;   //This is test size ,Real size 600
const int NUM_PIXELS = WIDTH * HEIGHT;

// States
const int EMPTY = 0;
const int SAND = 1;


// GPU DEVICE CODE 


// 1. The Atomic Traffic Cop
// This replaces Unity's AtomicCheckUnclaimed function
__device__ bool AtomicCheckUnclaimed(unsigned int* claims, int targetIdx) {
    // Attempt to change the claim from 0 to 1. 
    // If it returns 0, we were the first ones here.
    unsigned int previous = atomicCAS(&claims[targetIdx], 0, 1);
    return (previous == 0);
}

// 2. The Compute Shader Kernel
// This replaces Unity's HandleSimulation and CSMain
// 2. The Compute Shader Kernel
// 2. The Compute Shader Kernel (Now with Mouse Repulsion!)
// Notice the 4 new arguments added to the end of this function
// 2. The Compute Shader Kernel (Upgraded Bulldozer!)
// 2. The Compute Shader Kernel (Eruption Bulldozer!)
__global__ void SimulateParticlesKernel(int* gridInput, int* gridOutput, unsigned int* claims, int width, int height, bool oddFrame, int mouseX, int mouseY, bool isRightMouseDown, int brushRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int currentIdx = y * width + x;
    int particle = gridInput[currentIdx];

    if (particle == EMPTY) return; 

    // --- THE ERUPTION BULLDOZER ---
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

            // Look up to 200 pixels away to find the surface!
            int maxBlastDistance = 200; 

            // Search OUTWAR starting from the particle, traveling through solid sand
            for (int step = 1; step < maxBlastDistance; step++) {
                int pushX = x + (int)(nx * step);
                int pushY = y + (int)(ny * step);
                
                if (pushX >= 0 && pushX < width && pushY >= 0 && pushY < height) {
                    int pushIdx = pushY * width + pushX;
                    
                    // The very first EMPTY spot of open air we break through to becomes our new home!
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
        
        // NEW: Only spawn sand if this specific pixel is empty!
        if (grid[y * width + x] == EMPTY) {
            float waveSpeed = 0.05f;
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

int main() {
    // --- 1. CUDA Memory Setup ---
    int *d_gridInput, *d_gridOutput;
    unsigned int *d_claims;
    uchar4 *d_colorBuffer;
    uchar4 *h_colorBuffer;

    cudaMalloc(&d_gridInput, NUM_PIXELS * sizeof(int));
    cudaMalloc(&d_gridOutput, NUM_PIXELS * sizeof(int));
    cudaMalloc(&d_claims, NUM_PIXELS * sizeof(unsigned int));
    cudaMalloc(&d_colorBuffer, NUM_PIXELS * sizeof(uchar4));
    
    h_colorBuffer = (uchar4*)malloc(NUM_PIXELS * sizeof(uchar4));

    cudaMemset(d_gridInput, 0, NUM_PIXELS * sizeof(int));
    cudaMemset(d_gridOutput, 0, NUM_PIXELS * sizeof(int));

    // --- 2. OpenGL & GLFW Setup ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Sand Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // Create an OpenGL Texture to display the colors
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);

    // --- 3. Grid/Block Setup ---
    dim3 threadsPerBlock(32, 8, 1);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    int simStep = 0;
    std::ofstream logFile("performance_log_32x8.csv");
    logFile << "Frame,SimTime(ms),RenderTime(ms),MemcpyTime(ms),TotalFrame(ms),FPS\n";

    // CUDA Events (reuse them every frame)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 4. Main Game Loop ---
    while (!glfwWindowShouldClose(window)) {
    auto frameStart = std::chrono::high_resolution_clock::now();

    glfwPollEvents(); 
    bool oddFrame = (simStep % 2 == 1);

    float simTime = 0.0f;
    float renderTime = 0.0f;
    float memcpyTime = 0.0f;

    // ---------------- INPUT PHASE ----------------
    if (isMouseDown) {
        int brushSize = 20;
        AddSandKernel<<<numBlocks, threadsPerBlock>>>(
            d_gridInput, (int)mouseX, (int)mouseY, brushSize, WIDTH, HEIGHT, simStep
        );
    }

    // ---------------- SIMULATION PHASE ----------------
    cudaMemcpy(d_gridOutput, d_gridInput, NUM_PIXELS * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(d_claims, 0, NUM_PIXELS * sizeof(unsigned int));

    cudaEventRecord(start);

    SimulateParticlesKernel<<<numBlocks, threadsPerBlock>>>(
        d_gridInput, d_gridOutput, d_claims,
        WIDTH, HEIGHT, oddFrame,
        (int)mouseX, (int)mouseY,
        isRightMouseDown, 20
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&simTime, start, stop);

    // ---------------- RENDER PHASE ----------------
    cudaEventRecord(start);

    RenderToColorKernel<<<numBlocks, threadsPerBlock>>>(
        d_gridOutput, d_colorBuffer, WIDTH, HEIGHT
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&renderTime, start, stop);

    // ---------------- MEMCPY TIMING ----------------
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_colorBuffer, d_colorBuffer,
               NUM_PIXELS * sizeof(uchar4),
               cudaMemcpyDeviceToHost);

    auto t2 = std::chrono::high_resolution_clock::now();

    memcpyTime = std::chrono::duration<float, std::milli>(t2 - t1).count();

    // ---------------- RENDER TO SCREEN ----------------
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, h_colorBuffer);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glfwSwapBuffers(window);

    // ---------------- FRAME TIME ----------------
    auto frameEnd = std::chrono::high_resolution_clock::now();
    float totalFrameTime =
        std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();

    float fps = 1000.0f / totalFrameTime;

    // ---------------- LOGGING ----------------
    if (simStep % 10 == 0) { // log every 10 frames
        logFile << simStep << ","
                << std::fixed << std::setprecision(4)
                << simTime << ","
                << renderTime << ","
                << memcpyTime << ","
                << totalFrameTime << ","
                << fps << "\n";
    }

    // ---------------- SWAP BUFFERS ----------------
    int* temp = d_gridInput;
    d_gridInput = d_gridOutput;
    d_gridOutput = temp;

    simStep++;
}

    //  5. Cleanup
    cudaFree(d_gridInput);
    cudaFree(d_gridOutput);
    cudaFree(d_claims);
    cudaFree(d_colorBuffer);
    free(h_colorBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    logFile.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}