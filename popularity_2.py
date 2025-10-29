from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.linalg import DenseVector
import math
from Method2 import compute_importance_method2
from query_generator import generate_query_dicts,get_column_stats

import os
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.16/libexec/openjdk.jdk/Contents/Home"
os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/4.0.1_1/libexec"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]

# -------------------------------
# Cosine similarity function
# -------------------------------
def cosine_similarity(v1, v2):
    dot = float(v1.dot(v2))
    norm1 = math.sqrt(v1.dot(v1))
    norm2 = math.sqrt(v2.dot(v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

# -------------------------------
# Compute importance based on your equation
# -------------------------------
def compute_importance_in_batches(df, pop_col="popularity", id_col="Unique_ID", batch_size=1000):
    numeric_cols = [c for c in df.columns if c not in [id_col, pop_col]]

    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df = assembler.transform(df).select(id_col, pop_col, "features")

    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2)
    df = normalizer.transform(df)

    df_local = df.collect()
    total = len(df_local)
    results = []

    for i in range(0, total, batch_size):
        batch = df_local[i:i + batch_size]
        for row_i in batch:
            t_i = DenseVector(row_i["norm_features"])
            pop_i = float(row_i[pop_col])
            id_i = row_i[id_col]

            # Compute Σ(1 - sim(t_i, t_k)) over all k ≠ i
            sim_sum = 0.0
            for row_k in df_local:
                if row_k[id_col] == id_i:
                    continue
                t_k = DenseVector(row_k["norm_features"])
                sim_sum += (1 - cosine_similarity(t_i, t_k))

            imp_t = pop_i * sim_sum
            results.append((id_i, imp_t))

    # Compute average importance of the set R'
    total_importance = sum([imp for _, imp in results]) / total if total > 0 else 0

    spark = SparkSession.builder.getOrCreate()
    result_df = spark.createDataFrame(results, [id_col, "importance"])
    result_df = result_df.withColumn("set_importance", result_df["importance"] * 0 + total_importance)

    return result_df

# -------------------------------
# Main run
# -------------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.appName("Importance_Equation").getOrCreate()

    # Example dataset (integer only)
    data = [
        (3, 4, 14, 24, 0.4),
        (3, 4, 14, 24, 0.4),
        (3, 4, 14, 24, 0.4),
        (3, 4, 14, 24, 0.4),
        (2, 5, 15, 25, 0.5),
        (3, 8, 18, 28, 0.6),
        (100, 200, 1200, 22, 0.50)
    ]
    columns = ["Unique_ID", "A", "B", "C", "popularity"]
    df = spark.createDataFrame(data, columns)

    col_stats = get_column_stats({"my_table": df})
    queries = generate_query_dicts(df, "my_table", col_stats, n_queries=10, max_conditions=1)
    result_df = compute_importance_method2(df, queries, feature_cols, T=None, id_col="Unique_ID", batch_size=batch_size)


    print("✅ Final importance per tuple:")
    result_df.show(truncate=False)
