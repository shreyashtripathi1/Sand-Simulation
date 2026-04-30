import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("performance_log_32x8.csv")

# ---------------- FPS vs Frame ----------------
plt.figure()
plt.plot(df["Frame"], df["FPS"])
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("FPS vs Frame")
plt.grid()
plt.savefig("fps_vs_frame.png")

# ---------------- Kernel Times ----------------
plt.figure()
plt.plot(df["Frame"], df["SimTime(ms)"], label="Simulation")
plt.plot(df["Frame"], df["RenderTime(ms)"], label="Render")
plt.plot(df["Frame"], df["MemcpyTime(ms)"], label="Memcpy")
plt.xlabel("Frame")
plt.ylabel("Time (ms)")
plt.title("Kernel Time vs Frame")
plt.legend()
plt.grid()
plt.savefig("kernel_time.png")

# ---------------- Total Frame Time ----------------
plt.figure()
plt.plot(df["Frame"], df["TotalFrame(ms)"])
plt.xlabel("Frame")
plt.ylabel("Time (ms)")
plt.title("Total Frame Time")
plt.grid()
plt.savefig("frame_time.png")

print("Graphs generated!")