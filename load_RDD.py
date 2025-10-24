from pyspark.sql import SparkSession
import os
import time
import pandas as pd
from datetime import datetime
from query_generator import get_column_stats, generate_query_dicts
from popularity import compute_popularity_in_batches
# from popularity_2 import compute_importance_in_batches_pop2  # Optional if you switch later

# ------------------ ENVIRONMENT SETUP ------------------
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# ------------------ INITIALIZE SPARK ------------------
spark = SparkSession.builder \
    .appName("QueryGeneratorWithDistribution") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.local.dir", "/tmp/spark_tmp") \
    .getOrCreate()

# ------------------ EXPERIMENT SETTINGS ------------------
dataset_sizes = ["1K", "10K", "50K", "100K", "150K", "200K", "500K", "1M"]
dataset_sizes = ["1K"]

query_counts = [100, 1000, 2000, 5000, 10000]
query_counts = [10000]

max_conditions = 1
batch_size = 100
base_path = "/Users/youssefbenmansour/Desktop/Master/period 5/Data Intensive System/project/data"

# Log file path
log_path = "run_log.csv"

# ------------------ MAIN EXPERIMENT LOOP ------------------
for size in dataset_sizes:
    csv_path = os.path.join(base_path, f"dataset_{size}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {csv_path} — file not found.")
        continue

    # Load dataset
    print(f" Loading dataset: {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.cache()
    df.createOrReplaceTempView("my_table")

    # Compute column stats once per dataset
    relational_db = {"my_table": df}
    col_stats = get_column_stats(relational_db)

    # Run for each number of queries
    for n_queries in query_counts:
        print(f"Running for dataset={size}, n_queries={n_queries} ...")

        start_time = time.time()

        # Generate queries
        queries = generate_query_dicts(df, "my_table", col_stats, n_queries=n_queries, max_conditions=max_conditions)

        # Compute popularity
        pop_df = compute_popularity_in_batches(df, queries, id_col="Unique_ID", batch_size=batch_size)

        # Trigger computation
        pop_df.orderBy("pop", ascending=False).show(5, truncate=False)

        # Compute runtime
        elapsed_seconds = round(time.time() - start_time, 2)
        print(f"Completed: {elapsed_seconds} seconds")

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
