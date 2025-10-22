from pyspark.sql import SparkSession
import os
from query_generator import get_column_stats, generate_query_dicts
from popularity import compute_popularity

# ------------------ Environment ------------------
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# ------------------ STEP 1: Initialize Spark ------------------
spark = SparkSession.builder \
    .appName("QueryGeneratorWithDistribution") \
    .getOrCreate()

# ------------------ STEP 2: Load dataset ------------------
csv_path = "data/dataset_1K.csv"  # change path
df = spark.read.csv(csv_path, header=True, inferSchema=True)
table_name = "my_table"
df.createOrReplaceTempView(table_name)

# ------------------ STEP 3: Compute column stats ------------------
relational_db = {table_name: df}
col_stats = get_column_stats(relational_db)

# ------------------ STEP 4: Generate query dictionaries ------------------
queries = generate_query_dicts(df, table_name, col_stats, n_queries=500, max_conditions=5)


# ------------------ STEP 5: Compute popularity ------------------

pop_df = compute_popularity(df, queries, id_col="patient_BSN")

# ------------------ STEP 6: Show results ------------------
pop_df.orderBy("pop", ascending=False).show(10)

