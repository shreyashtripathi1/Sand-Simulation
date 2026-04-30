import pandas as pd
import matplotlib.pyplot as plt

# Load files
df_8 = pd.read_csv("performance_log_8x8.csv")
df_16 = pd.read_csv("performance_log_16x16.csv")
df_32 = pd.read_csv("performance_log_32x8.csv")

# ---------------- FPS vs Frame ----------------
plt.figure()
plt.plot(df_8["Frame"], df_8["FPS"], label="8x8")
plt.plot(df_16["Frame"], df_16["FPS"], label="16x16")
plt.plot(df_32["Frame"], df_32["FPS"], label="32x8")

plt.xlabel("Frame")
plt.ylabel("FPS")
plt.title("FPS Comparison (Block Sizes)")
plt.legend()
plt.grid()
plt.savefig("fps_comparison.png")

# ---------------- Frame Time ----------------
plt.figure()
plt.plot(df_8["Frame"], df_8["TotalFrame(ms)"], label="8x8")
plt.plot(df_16["Frame"], df_16["TotalFrame(ms)"], label="16x16")
plt.plot(df_32["Frame"], df_32["TotalFrame(ms)"], label="32x8")

plt.xlabel("Frame")
plt.ylabel("Time (ms)")
plt.title("Frame Time Comparison")
plt.legend()
plt.grid()
plt.savefig("frame_time_comparison.png")

# ---------------- Average Metrics ----------------
avg_8 = df_8["FPS"].mean()
avg_16 = df_16["FPS"].mean()
avg_32 = df_32["FPS"].mean()

print("\n===== AVERAGE FPS =====")
print(f"8x8   : {avg_8:.2f}")
print(f"16x16 : {avg_16:.2f}")
print(f"32x8  : {avg_32:.2f}")

# ---------------- Best Configuration ----------------
best = max(avg_8, avg_16, avg_32)

print("\n===== BEST CONFIG =====")
if best == avg_16:
    print("16x16 is BEST 🚀")
elif best == avg_32:
    print("32x8 is BEST 🚀")
else:
    print("8x8 is BEST 🚀")

print("\nGraphs saved as:")
print(" - fps_comparison.png")
print(" - frame_time_comparison.png")