from functools import reduce
from pyspark.sql import functions as F


def compute_popularity(df, queries, id_col="Unique_ID"):
    if not queries:
        return df.select(F.col(id_col).alias("id")).withColumn("pop", F.lit(0))

    and_conds = []
    for q in queries:
        if not q:
            continue
        cond = reduce(
            lambda a, b: a & b,
            (F.col(c).eqNullSafe(F.lit(v)) for c, v in q.items())
        )
        and_conds.append(cond)

    if not and_conds:
        return df.select(F.col(id_col).alias("id")).withColumn("pop", F.lit(0))

    pop_incr = None
    for cond in and_conds:
        term = F.when(cond, F.lit(1)).otherwise(F.lit(0))
        pop_incr = term if pop_incr is None else (pop_incr + term)

    return (
        df.withColumn("pop", pop_incr)
          .groupBy(F.col(id_col).alias("id"))
          .agg(F.sum("pop").alias("pop"))
    )

def compute_popularity_in_batches(df, query_dicts, id_col="Patient_BSN", batch_size=50):
    total_pop = None

    for i in range(0, len(query_dicts), batch_size):
        batch = query_dicts[i:i+batch_size]
        batch_pop = compute_popularity(df, batch, id_col=id_col)
        batch_pop = batch_pop.withColumnRenamed("pop", "batch_pop")  # rename to avoid ambiguity

        if total_pop is None:
            total_pop = batch_pop.withColumnRenamed("batch_pop", "pop")
        else:
            total_pop = total_pop.join(batch_pop, on="id", how="outer").fillna(0)
            total_pop = total_pop.withColumn("pop", F.col("pop") + F.col("batch_pop")).drop("batch_pop")

    return total_pop.orderBy(F.col("pop").desc())