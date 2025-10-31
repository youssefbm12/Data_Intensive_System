from pyspark.sql import SparkSession
import os
from method3_PAM import method3_pam_full
from method3_SpPam import method3_pam_spark
from query_generator import get_column_stats, generate_query_dicts
from popularity import compute_popularity_in_batches
import time

from Method3 import method3_kmedoids_clara
from pyspark.sql import functions as F, types as T
from method3_kcenter_sharded import method3_kcenter_sharded
# ------------------ Timer ------------------
start_time = time.time()

import os
import os, shutil, sys, platform

# 1) Pin Java & Python for Spark (must be before importing pyspark)
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["PATH"]      = os.environ["JAVA_HOME"] + r"\bin;" + os.environ.get("PATH","")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.pop("SPARK_HOME", None)   # avoid pointing at a wrong install
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"  # avoids odd bind issues on Windows

# (Optional) If you installed Spark separately, point to it
# os.environ["SPARK_HOME"] = r"C:\tools\spark-3.5.1-bin-hadoop3"  # <- if you have it
# os.environ["PATH"] = os.environ["SPARK_HOME"] + r"\bin;" + os.environ["PATH"]

# --- Sanity checks: fail fast with actionable messages ---
java_path = shutil.which("java")
if not java_path:
    raise RuntimeError(
        "Java not found on PATH. Install JDK 17 (Adoptium Temurin) and set JAVA_HOME.\n"
        "Example JAVA_HOME: C:\\Program Files\\Eclipse Adoptium\\jdk-17\n"
        "Then reopen terminal or set os.environ as above."
    )

print("Python:", platform.python_version())
print("Java:", java_path)



# ------------------ STEP 1: Initialize Spark ------------------
spark = SparkSession.builder \
    .appName("QueryGeneratorWithDistribution") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.shuffle.partitions", "96")   \
    .config("spark.local.dir", "/tmp/spark_tmp") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()


# ------------------ STEP 2: Load dataset ------------------
ID_COL = "Unique_ID"
csv_path = "dataset_1K.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)
table_name = "my_table"
df.createOrReplaceTempView(table_name)

int_types = (T.IntegerType, T.LongType, T.ShortType, T.ByteType)
numeric_cols = [c for c, dt in df.dtypes if c != ID_COL and isinstance(df.schema[c].dataType, int_types)]

# Cache the DataFrame to reduce recomputation and shuffle issues
df.cache()

# ------------------ STEP 3: Compute column stats ------------------
relational_db = {table_name: df}
col_stats = get_column_stats(relational_db)

# ------------------ STEP 4: Generate query dictionaries ------------------
# Use smaller batches if needed to avoid memory errors
queries = generate_query_dicts(df, table_name, col_stats, n_queries=8000, max_conditions=5)
print(f"Generated {len(queries)} queries.")
# ------------------ STEP 5: Compute popularity in batches ------------------
# Here we batch queries to avoid Spark shuffle/broadcast errors
batch_size = 50  # adjust depending on memory
#pop_df = compute_popularity_in_batches(df, queries, id_col="Unique_ID", batch_size=batch_size)

T_keep = 15
#######K-CLARA
# best_ids = method3_kmedoids_clara(
#     df,
#     id_col=ID_COL,
#     numeric_cols=numeric_cols,   # integers (cast to DOUBLE internally)
#     t_keep=T_keep,
#     sample_size=50000,           # 20k–50k for ~1M rows
#     restarts=8,                  # 5–10 restarts
#     pca_k=55,                     # set to 100 if you have many columns
#     seed=42
# )
# #######K-CENTER SHARDED
# R_prime = method3_kcenter_sharded(
#     df,
#     id_col=ID_COL,
#     numeric_cols=numeric_cols,
#     t_keep=100,        # your target
#     shards=100,            # 1,000 picks per shard
#     per_shard_sample=10000,  # ~10k sampled/collected per shard (tune 5k–20k)
#     seed=42
# )
#######PAM-NumPy 
# best_ids = method3_pam_full(
#     df,
#     id_col=ID_COL,
#     numeric_cols=numeric_cols,
#     t_keep=T_keep,
#     seed=42,                
#     cast_features_to_double=True
# )
best_ids = method3_pam_spark(
    df,
    id_col=ID_COL,
    numeric_cols=numeric_cols,
    t_keep=T_keep,           
    pca_k=64,           
    seed=42,
    max_iter=20,
    assign_batch=64    
)

print(best_ids[:10])
# ------------------ STEP 6: Show top results ------------------
#pop_df.orderBy("pop", ascending=False).show(10)

# ------------------ STEP 7: Timer ------------------
end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Time taken: {elapsed_seconds:.2f} seconds")


#print(pop_df.head(10))
