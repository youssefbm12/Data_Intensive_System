from functools import reduce
from pyspark.sql import functions as F


def compute_popularity(df, queries, id_col="patient_BSN"):
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