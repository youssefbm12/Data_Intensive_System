# method3_SpPam.py
# Pure Spark k-Medoids (PAM) in cosine space — NO NumPy, NO Python UDFs.

from __future__ import annotations
from typing import List, Tuple, Optional

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, types as T
from pyspark.ml.feature import VectorAssembler, Normalizer, PCA as MLPCA
from pyspark.ml.functions import vector_to_array

# ---------- SQL array helpers ----------
def _dot_with_medoid_expr(med: List[float]) -> F.Column:
    arr = ",".join(f"CAST({float(v)} AS DOUBLE)" for v in med)
    return F.expr("aggregate(zip_with(_arr, array(" + arr + "), (a,b)->a*b), 0D, (acc,x)->acc+x)")

def _elementwise_sum_arrays(col_arr: F.Column) -> F.Column:
    # init must be typed: array<double>
    init = F.array().cast(T.ArrayType(T.DoubleType()))
    return F.aggregate(
        F.collect_list(col_arr),
        init,
        lambda acc, x: F.when(F.size(acc) == 0, x).otherwise(F.zip_with(acc, x, lambda a, b: a + b))
    )

def _argmax_index(cols: List[str]) -> F.Column:
    mx = F.greatest(*[F.col(c) for c in cols])
    expr = F.when(F.col(cols[0]) == mx, F.lit(0))
    for j in range(1, len(cols)):
        expr = expr.when(F.col(cols[j]) == mx, F.lit(j))
    return expr

# ---------- Seeding: greedy farthest-point on Spark ----------
def _farthest_point_seeding_spark(df: DataFrame, k: int, seed: int) -> Tuple[List[List[float]], List]:
    first = df.orderBy(F.rand(seed)).limit(1).select("*").first()
    med_vecs, med_ids = [first["_arr"]], [first[0]]  # assumes id_col is first selected col
    work = df.withColumn("_best_sim", _dot_with_medoid_expr(med_vecs[0]))
    for _ in range(1, k):
        far = work.orderBy(F.col("_best_sim").asc()).limit(1).select("*").first()
        med_vecs.append(far["_arr"]); med_ids.append(far[0])
        work = work.withColumn("_best_sim", F.greatest(F.col("_best_sim"), _dot_with_medoid_expr(far["_arr"])))
    return med_vecs, med_ids

# ---------- Assignment with batching (prevents gigantic plans) ----------
def _assign_to_medoids_batched(df: DataFrame, med_vecs: List[List[float]], batch: Optional[int]) -> DataFrame:
    if not batch or batch <= 0:
        names = [f"_s{j}" for j in range(len(med_vecs))]
        sims = [_dot_with_medoid_expr(m).alias(n) for m, n in zip(med_vecs, names)]
        tmp = df.select("*", *sims)
        return tmp.withColumn("_cid", _argmax_index(names)).drop(*names)

    # build sims in chunks, then compute global argmax (second pass keeps code simple)
    names_all, tmp = [], df
    idx = 0
    for i in range(0, len(med_vecs), batch):
        chunk = med_vecs[i:i+batch]
        names = [f"_s{idx+j}" for j in range(len(chunk))]
        sims = [_dot_with_medoid_expr(m).alias(n) for m, n in zip(chunk, names)]
        tmp = tmp.select("*", *sims)
        names_all.extend(names); idx += len(chunk)
    out = tmp.withColumn("_cid", _argmax_index(names_all)).drop(*names_all)
    return out

# --- scalable elementwise pos explode ---
def _sumvec_by_cluster(df_assigned: DataFrame) -> DataFrame:
    # df_assigned: columns ["_cid", id_col, "_arr"]
    exploded = df_assigned.select(
        F.col("_cid"),
        F.posexplode(F.col("_arr")).alias("_idx", "_val")
    )
    summed = (exploded
              .groupBy("_cid", "_idx")
              .agg(F.sum("_val").alias("_s")))
    # collect back per cluster in index order -> array<double>
    pairs = (summed.groupBy("_cid")
             .agg(F.sort_array(F.collect_list(F.struct("_idx", "_s"))).alias("_pairs")))
    sumvec = pairs.select(
        "_cid",
        F.transform(F.col("_pairs"), lambda p: p["_s"]).alias("_sumvec")
    )
    return sumvec

# ---------- Update medoids (exact, scalar-only shuffle) ----------
def _update_medoids_cosine(df_assigned: DataFrame, id_col: str):
    # df_assigned has: ["_cid", id_col, "_arr"]

    # 1) cluster sums (array<double>) — we won't shuffle arrays later
    sums = _sumvec_by_cluster(df_assigned)  # ["_cid", "_sumvec"]

    # 2) score each point; then immediately drop arrays to keep shuffle payload tiny
    scored_small = (
        df_assigned.join(sums, "_cid")
        .withColumn(
            "_score",
            F.expr("aggregate(zip_with(_arr, _sumvec, (a,b)->a*b), 0D, (acc,x)->acc+x)")
        )
        .select("_cid", id_col, "_score")    # drop _arr/_sumvec before groupBy/shuffle
        .repartition(F.col("_cid"))          # spread clusters; prevents skew OOM
    )
    # Optional lineage cut (enable if local heap is tight):
    #scored_small = scored_small.localCheckpoint(eager=True)    

    # 3) max score per cluster (scalar)
    max_scores = scored_small.groupBy("_cid").agg(F.max("_score").alias("_m"))

    # 4) rows that hit the max (ties remain); still scalar only
    tops_ids = (
        scored_small.join(max_scores, "_cid")
        .where(F.col("_score") == F.col("_m"))
        .select("_cid", id_col)
    )

    # 5) deterministic tie-break: smallest id
    winners_ids = tops_ids.groupBy("_cid").agg(F.min(id_col).alias(id_col))

    # 6) fetch vectors ONLY for the k winners; join on both keys for safety
    winners = (
        winners_ids
        .join(df_assigned.select("_cid", id_col, "_arr"), ["_cid", id_col], "inner")
        .orderBy("_cid")
        .select("_cid", id_col, "_arr")
        .coalesce(1)                         # tiny result
    )

    rows = winners.collect()                 # k rows only
    new_ids  = [r[id_col] for r in rows]
    new_vecs = [r["_arr"]   for r in rows]
    return new_ids, new_vecs

# ---------- Public API ----------
def method3_pam_spark(
    df: DataFrame, *,
    id_col: str,
    numeric_cols: List[str],
    t_keep: int,
    pca_k: int = 0,
    seed: int = 42,
    max_iter: int = 20,
    cast_features_to_double: bool = True,
    assign_batch: Optional[int] = 256,
) -> List:
    """
    Pure Spark k-Medoids (PAM) under cosine distance (1 - cosine similarity).
    Exact updates; shuffles only scalars for stability. Returns: list of medoid IDs.

    Notes:
    - Set a checkpoint dir once in your session: sc.setCheckpointDir("file:///C:/temp/spark_checkpoint")
    - Prefer calling with pca_k=64 and assign_batch=64~128 for wide dims / large k.
    """
    assert t_keep > 0, "t_keep must be positive"
    if id_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != id_col]

    # Cast features
    if cast_features_to_double:
        for c in numeric_cols:
            df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Assemble -> (optional) PCA -> normalize -> array
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="_raw_vec")
    base = assembler.transform(df).select(id_col, "_raw_vec").dropna(subset=["_raw_vec"])

    if pca_k and pca_k > 0:
        k_eff = min(pca_k, len(numeric_cols))
        pca = MLPCA(k=k_eff, inputCol="_raw_vec", outputCol="_feat_vec")
        model = pca.fit(base)
        vdf = model.transform(base).select(id_col, "_feat_vec").withColumnRenamed("_feat_vec", "_feat")
    else:
        vdf = base.withColumnRenamed("_raw_vec", "_feat")

    norm = Normalizer(inputCol="_feat", outputCol="_norm", p=2.0)
    fdf = norm.transform(vdf).select(id_col, vector_to_array(F.col("_norm")).alias("_arr"))

    # Drop zero vectors
    sq = F.aggregate(F.transform(F.col("_arr"), lambda x: x*x), F.lit(0.0), lambda acc, x: acc + x)
    fdf = fdf.withColumn("_sq", sq).where(F.col("_sq") > 0.0).drop("_sq")

    # Balance partitions + allow spill
    target_parts = max(96, df.sparkSession.sparkContext.defaultParallelism * 2)
    fdf = fdf.repartition(target_parts, F.col(id_col)) \
             .persist(StorageLevel.MEMORY_AND_DISK)
    n = fdf.count()
    if n == 0:
        return []
    if t_keep > n:
        raise ValueError(f"t_keep ({t_keep}) must be <= rows ({n})")

    # Seeding
    med_vecs, med_ids = _farthest_point_seeding_spark(fdf.select(id_col, "_arr"), t_keep, seed)

    # PAM loop
    for _ in range(max_iter):
        assigned = _assign_to_medoids_batched(
            fdf.select(id_col, "_arr"), med_vecs, assign_batch
        )
        new_ids, new_vecs = _update_medoids_cosine(assigned, id_col)
        if new_ids == med_ids:
            break
        med_ids, med_vecs = new_ids, new_vecs
        # free cached columns/plans between iters (helps local[*] heap)
        df.sparkSession.catalog.clearCache()

    return med_ids
