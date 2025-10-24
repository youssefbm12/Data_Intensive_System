from pyspark.sql import functions as F

def compute_popularity_fast(df, queries, id_col="Unique_ID"):
    """
    Compute popularity using a more vectorized approach.
    Each query is converted into a filter, and we count matches efficiently.
    """
    if not queries:
        return df.select(F.col(id_col).alias("id")).withColumn("pop", F.lit(0))

    # Initialize a column for popularity
    df = df.withColumn("pop", F.lit(0))

    for q in queries:
        # Build the AND condition directly for this query
        cond = F.lit(True)
        for col, val in q.items():
            cond = cond & (F.col(col) == F.lit(val))
        
        # Increment popularity only for matching rows
        df = df.withColumn("pop", F.when(cond, F.col("pop") + 1).otherwise(F.col("pop")))

    return df.select(F.col(id_col).alias("id"), "pop")

def compute_popularity_in_batches(df, query_dicts, id_col="Unique_ID", batch_size=200):
    """
    Batch computation that avoids excessive joins.
    Aggregates within Spark instead of repeated DataFrame joins.
    """
    total_pop = None

    for i in range(0, len(query_dicts), batch_size):
        batch = query_dicts[i:i+batch_size]
        batch_pop = compute_popularity_fast(df, batch, id_col=id_col)

        if total_pop is None:
            total_pop = batch_pop
        else:
            # Use a single join and aggregation instead of multiple joins
            total_pop = total_pop.union(batch_pop)

    # Aggregate popularity for all batches
    total_pop = total_pop.groupBy("id").agg(F.sum("pop").alias("pop"))

    return total_pop.orderBy(F.col("pop").desc())
