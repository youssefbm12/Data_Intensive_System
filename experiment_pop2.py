import os
import time
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from query_generator import get_column_stats, generate_query_dicts
from Method2 import compute_importance_method2  # import your method
from pyspark.sql import types as T

# ------------------ Environment ------------------
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# ------------------ Initialize Spark ------------------
spark = SparkSession.builder \
    .appName("ImportanceExperiment") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.local.dir", "/tmp/spark_tmp") \
    .getOrCreate()

# ------------------ Experiment Settings ------------------
subset_sizes = [50000]  # how many rows to take
query_counts = [100, 1000, 2000, 5000, 10000]
max_conditions = 1
batch_size = 100

log_path = "run_log_method2.csv"
dataset_path = "data/dataset_1M.csv"  # your full dataset (1M rows)

# ------------------ Load Dataset Once ------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ File not found: {dataset_path}")

print(f"Loading full dataset: {dataset_path}")
df_full = spark.read.csv(dataset_path, header=True, inferSchema=True).cache()
df_full.createOrReplaceTempView("my_table_full")

# ------------------ Column Stats ------------------
col_stats = get_column_stats({"my_table": df_full})

# ------------------ Main Experiment Loop ------------------
for n_rows in subset_sizes:
    print(f"\n🧩 Using first {n_rows:,} rows...")
    df = df_full.limit(n_rows)
    df.cache()
    df.createOrReplaceTempView("my_table")

    # Select numeric feature columns
    feature_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (T.IntegerType, T.FloatType, T.DoubleType, T.LongType))
        and f.name != "Unique_ID"
    ]

    for n_queries in query_counts:
        print(f"⚙️ Running for n_rows={n_rows}, n_queries={n_queries} ...")
        start_time = time.time()

        # Generate synthetic queries
        queries = generate_query_dicts(df, "my_table", col_stats, n_queries=n_queries, max_conditions=max_conditions)

        # Compute importance using your method
        imp_df = compute_importance_method2(
            df, queries, feature_cols,
            T=None, id_col="Unique_ID", batch_size=batch_size
        )

        # Trigger computation (to force evaluation)
        imp_df.show(5, truncate=False)

        elapsed_seconds = round(time.time() - start_time, 2)
        print(f"✅ Completed in {elapsed_seconds} seconds")

        # ------------------ Logging ------------------
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": n_rows,
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
        print(f"📝 Log updated for {n_rows} rows, {n_queries} queries.")

print("\n🎯 All experiments completed successfully!")
