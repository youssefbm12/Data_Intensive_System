from pyspark.sql import SparkSession
import os
from query_generator import get_column_stats, generate_query_dicts
from popularity import compute_popularity_in_batches
import time

# ------------------ Timer ------------------
start_time = time.time()

# ------------------ Environment ------------------
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# ------------------ STEP 1: Initialize Spark ------------------
spark = SparkSession.builder \
    .appName("QueryGeneratorWithDistribution") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.local.dir", "/tmp/spark_tmp") \
    .getOrCreate()

# ------------------ STEP 2: Load dataset ------------------
csv_path = "dataset_1K.csv"

df = spark.read.csv(csv_path, header=True, inferSchema=True)
table_name = "my_table"
df.createOrReplaceTempView(table_name)

# Cache the DataFrame to reduce recomputation and shuffle issues
df.cache()

# ------------------ STEP 3: Compute column stats ------------------
relational_db = {table_name: df}
col_stats = get_column_stats(relational_db)

# ------------------ STEP 4: Generate query dictionaries ------------------
# Use smaller batches if needed to avoid memory errors
queries = generate_query_dicts(df, table_name, col_stats, n_queries=100, max_conditions=1)

# ------------------ STEP 5: Compute popularity in batches ------------------
# Here we batch queries to avoid Spark shuffle/broadcast errors
batch_size = 50  # adjust depending on memory
pop_df = compute_popularity_in_batches(df, queries, id_col="Unique_ID", batch_size=batch_size)

# ------------------ STEP 6: Show top results ------------------
pop_df.orderBy("pop", ascending=False).show(10)

# ------------------ STEP 7: Timer ------------------
end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Time taken: {elapsed_seconds:.2f} seconds")
