from pyspark.sql import SparkSession
import os
from query_generator import get_column_stats, generate_query_dicts
from popularity import compute_popularity_in_batches
import time

from Method3 import method3_kmedoids_clara
from pyspark.sql import functions as F, types as T

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
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.local.dir", "/tmp/spark_tmp") \
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

T_keep = 1000  #Tuples to keep 

best_ids = method3_kmedoids_clara(
    df,
    id_col=ID_COL,
    numeric_cols=numeric_cols,   # integers (will be cast to DOUBLE internally)
    t_keep=T_keep,
    sample_size=50000,           # 20k–50k for ~1M rows
    restarts=8,                  # 5–10 restarts
    pca_k=55,                    # set to max columns
    seed=42
)

R_prime = df.where(F.col(ID_COL).isin(best_ids)).dropDuplicates([ID_COL])

print(R_prime.head(10))
# ------------------ STEP 6: Show top results ------------------
#pop_df.orderBy("pop", ascending=False).show(10)

# ------------------ STEP 7: Timer ------------------
end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Time taken: {elapsed_seconds:.2f} seconds")


#print(pop_df.head(10))
