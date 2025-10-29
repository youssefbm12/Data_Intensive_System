from pyspark.sql import SparkSession
import os
import time
import pandas as pd
from datetime import datetime
from query_generator import get_column_stats, generate_query_dicts
from Method2 import method2  

# ------------------ ENVIRONMENT SETUP ------------------
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# ------------------ INITIALIZE SPARK ------------------
spark = SparkSession.builder \
    .appName("Method2Experiment") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.local.dir", "/tmp/spark_tmp") \
    .getOrCreate()

# ------------------ EXPERIMENT SETTINGS ------------------
dataset_sizes = ["1K", "10K", "50K", "100K", "150K", "200K", "500K", "1M"]
dataset_sizes = ["1K"]

query_counts = [100, 1000, 2000, 5000, 10000]
query_counts = [3]

max_conditions = 1
T = 10  # number of tuples to select
base_path = ""

# Log file path
log_path = "run_log.csv"

# ------------------ MAIN EXPERIMENT LOOP ------------------
for size in dataset_sizes:
    csv_path = os.path.join(base_path, f"dataset_{size}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {csv_path} — file not found.")
        continue

    # Load dataset
    print(f"📂 Loading dataset: {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.cache()
    df.createOrReplaceTempView("my_table")

    # Compute column stats once per dataset
    relational_db = {"my_table": df}
    col_stats = get_column_stats(relational_db)

    feature_cols = [c for c in df.columns if c != "Unique_ID"]

    # Run for each number of queries
    for n_queries in query_counts:
        print(f"Running Method2 for dataset={size}, n_queries={n_queries} ...")

        start_time = time.time()

        # Generate queries
        queries = generate_query_dicts(
            df, "my_table", col_stats, n_queries=n_queries, max_conditions=max_conditions
        )

        # ✅ Run new method
        result_df = method2(
            spark=spark,
            df=df,
            queries=queries,
            T=T,
            feature_cols=feature_cols,
            id_col="Unique_ID"
        )

        # Trigger computation and preview
        print("🔹 Top results:")
        result_df.show(5, truncate=False)

        # Compute runtime
        elapsed_seconds = round(time.time() - start_time, 2)
        print(f"Completed in {elapsed_seconds} seconds")

        # ------------------ LOGGING ------------------
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": size,
            "n_queries": n_queries,
            "max_conditions": max_conditions,
            "time_seconds": elapsed_seconds
        }

        if os.path.exists(log_path):
            existing_log = pd.read_csv(log_path)
            updated_log = pd.concat([existing_log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            updated_log = pd.DataFrame([log_entry])

        updated_log.to_csv(log_path, index=False)
        print(f"Log updated for dataset={size}, queries={n_queries}")

print("All experiments completed successfully!")
