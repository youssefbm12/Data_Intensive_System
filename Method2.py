from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import types as T
from pyspark.ml.functions import vector_to_array


# ------------------ Popularity ------------------
def compute_popularity(df, queries, id_col="Unique_ID"):
    if not queries:
        return df.select(F.col(id_col).alias("id")).withColumn("pop", F.lit(0))

    and_conds = []
    for q in queries:
        if not q:
            continue
        cond = None
        for c, v in q.items():
            term = F.col(c).eqNullSafe(F.lit(v))
            cond = term if cond is None else (cond & term)
        and_conds.append(cond)

    if not and_conds:
        return df.select(F.col(id_col).alias("id")).withColumn("pop", F.lit(0))

    pop_expr = None
    for cond in and_conds:
        term = F.when(cond, 1).otherwise(0)
        pop_expr = term if pop_expr is None else (pop_expr + term)

    return (
        df.withColumn("pop", pop_expr)
          .groupBy(F.col(id_col).alias("id"))
          .agg(F.sum("pop").alias("pop"))
    )


def compute_popularity_in_batches(df, query_dicts, id_col="Unique_ID", batch_size=50):
    total_pop = None
    for i in range(0, len(query_dicts), batch_size):
        batch = query_dicts[i:i+batch_size]
        batch_pop = compute_popularity(df, batch, id_col=id_col).withColumnRenamed("pop", "batch_pop")

        if total_pop is None:
            total_pop = batch_pop.withColumnRenamed("batch_pop", "pop")
        else:
            total_pop = (
                total_pop.join(batch_pop, on="id", how="outer")
                .fillna(0)
                .withColumn("pop", F.col("pop") + F.col("batch_pop"))
                .drop("batch_pop")
            )
    return total_pop


# ------------------ Fully Spark-Parallel Importance ------------------
def compute_importance_method2(df, queries, feature_cols, id_col="Unique_ID", batch_size=50, T=None):
    # --- Step 1: Popularity ---
    pop_df = compute_popularity_in_batches(df, queries, id_col=id_col, batch_size=batch_size)

    # --- Step 2: Assemble features ---
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df).select(id_col, "features")

    # Convert VectorUDT → ArrayType(DoubleType)
    df_vec = df_vec.withColumn("features_arr", vector_to_array("features"))


    # --- Step 3: Normalize features (Spark-only) ---
    df_vec = df_vec.withColumn(
        "norm",
        F.sqrt(
            F.aggregate(
                F.expr("transform(features_arr, x -> x * x)"),
                F.lit(0.0),
                lambda acc, x: acc + x
            )
        )
    )

    df_vec = df_vec.withColumn(
        "norm",
        F.when(F.col("norm") == 0, F.lit(1e-9)).otherwise(F.col("norm"))
    )

    df_vec = df_vec.withColumn(
        "norm_features",
        F.expr("transform(features_arr, x -> x / norm)")
    )

    # --- Step 4: Pairwise cosine similarities via cross join ---
    cross = (
        df_vec.alias("a")
        .crossJoin(df_vec.alias("b"))
        .filter(F.col(f"a.{id_col}") != F.col(f"b.{id_col}"))
    )

    cross = cross.withColumn(
        "cosine",
        F.aggregate(
            F.expr("zip_with(a.norm_features, b.norm_features, (x, y) -> x * y)"),
            F.lit(0.0),
            lambda acc, x: acc + x
        )
    )

    cross = cross.withColumn("dissim", 1 - F.col("cosine"))

    # --- Step 5: Average dissimilarity per tuple ---
    avg_dissim_df = (
        cross.groupBy(F.col(f"a.{id_col}").alias("id"))
             .agg(F.avg("dissim").alias("avg_dissim"))
    )

    # --- Step 6: Combine with popularity ---
    imp_df = (
        avg_dissim_df.join(pop_df, on="id", how="left")
                     .fillna(0, subset=["pop"])
                     .withColumn("importance", F.col("pop") * F.col("avg_dissim"))
                     .orderBy(F.col("importance").desc())
    )

    if T is not None:
        imp_df = imp_df.limit(T)

    return imp_df
