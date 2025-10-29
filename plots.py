import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Paste your log directly here
log_data = """2025-10-26 11:51:41,1000.0,100.0,1.0,1.52
2025-10-26 11:51:45,1000.0,1000.0,1.0,3.92
2025-10-26 11:51:51,1000.0,2000.0,1.0,6.86
2025-10-26 11:52:13,1000.0,5000.0,1.0,21.4
2025-10-26 11:53:17,1000.0,10000.0,1.0,64.16
2025-10-26 11:53:19,10000.0,100.0,1.0,1.56
2025-10-26 11:53:23,10000.0,1000.0,1.0,4.04
2025-10-26 11:53:30,10000.0,2000.0,1.0,7.31
2025-10-26 11:53:53,10000.0,5000.0,1.0,23.13
2025-10-26 11:55:02,10000.0,10000.0,1.0,69.12
2025-10-26 11:55:06,20000.0,100.0,1.0,3.95
2025-10-26 11:55:12,20000.0,1000.0,1.0,5.93
2025-10-26 11:55:22,20000.0,2000.0,1.0,10.3
2025-10-26 11:55:50,20000.0,5000.0,1.0,27.5
2025-10-26 11:57:03,20000.0,10000.0,1.0,72.95
2025-10-26 11:57:36,30000.0,100.0,1.0,33.49
2025-10-26 11:58:21,30000.0,1000.0,1.0,44.23
2025-10-26 11:59:02,30000.0,2000.0,1.0,41.81
2025-10-26 12:00:06,30000.0,5000.0,1.0,63.64
2025-10-26 12:01:55,30000.0,10000.0,1.0,108.87
2025-10-26 12:04:37,40000.0,100.0,1.0,162.37
2025-10-26 12:07:21,40000.0,1000.0,1.0,163.32
2025-10-26 12:09:51,40000.0,2000.0,1.0,149.89
2025-10-26 12:12:52,40000.0,5000.0,1.0,181.57
2025-10-26 12:16:39,40000.0,10000.0,1.0,226.65
2025-10-26 12:21:54,50000.0,100.0,1.0,314.89
2025-10-26 12:27:14,50000.0,1000.0,1.0,319.56
2025-10-26 12:33:01,50000.0,2000.0,1.0,347.04
2025-10-26 12:39:17,50000.0,5000.0,1.0,376.63
2025-10-26 12:46:37,50000.0,10000.0,1.0,439.75"""

# Load into a DataFrame
df = pd.read_csv(StringIO(log_data),
                 names=["timestamp", "dataset_size", "n_queries", "method", "time_seconds"])

# Convert numeric columns
df["dataset_size"] = df["dataset_size"].astype(float)
df["n_queries"] = df["n_queries"].astype(float)
df["time_seconds"] = df["time_seconds"].astype(float)

# Sort for clean plotting
df = df.sort_values(["n_queries", "dataset_size"])

# ------------------ Plot ------------------
plt.figure(figsize=(10, 6))

for n_queries, group in df.groupby("n_queries"):
    plt.plot(group["dataset_size"], group["time_seconds"],
             marker="o", label=f"{int(n_queries)} queries")

plt.title("Runtime Scaling vs Dataset Size", fontsize=14)
plt.xlabel("Dataset Size (rows)", fontsize=12)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Number of Queries", loc="upper left")
plt.tight_layout()

plt.show()
# plt.savefig("runtime_scaling.png", dpi=300)
