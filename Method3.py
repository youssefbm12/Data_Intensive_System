# method3.py
# Method 3: k-Medoids (CLARA) with NO Python UDFs (stable on Windows)
# - Integer feature columns are cast to DOUBLE, then vectorized, (optional PCA), L2-normalized.
# - PAM runs on a sample (driver, NumPy), best medoids evaluated on FULL data using SQL array ops.
# - Returns list of id values for the selected medoids (R').

from __future__ import annotations
from typing import List
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T  # keep this name; we don't use parameter T
from pyspark.ml.feature import VectorAssembler, PCA as MLPCA, Normalizer
from pyspark.ml.functions import vector_to_array

# ----------------- NumPy helpers (driver-side) -----------------

def _farthest_point_seeding(X: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """k-center style seeding in cosine space (X is L2-normalized)."""
    n = X.shape[0]
    idxs = [int(rng.integers(0, n))]
    sims = X @ X[idxs[0]]          # cosine sims to first
    d = 1.0 - sims                 # cosine distance
    for _ in range(1, k):
        i = int(np.argmax(d))
        idxs.append(i)
        sims = np.maximum(sims, X @ X[i])
        d = 1.0 - sims
    return idxs

def _pam_on_sample(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> List[int]:
    """
    PAM on a L2-normalized sample (X: n x d). Distance = 1 - dot.
    Seeding = farthest-point. Update = choose member maximizing sum of sims.
    Returns medoid indices w.r.t. X.
    """
    rng = np.random.default_rng(seed)
    medoids = _farthest_point_seeding(X, k, rng)

    for _ in range(max_iter):
        M = X[medoids]                  # (k, d)
        S = X @ M.T                     # (n, k) cosine sims
        assign = np.argmax(S, axis=1)
        changed, new_medoids = False, []

        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                # empty cluster: pick farthest from existing medoids
                sims_to_M = S.max(axis=1)
                far_idx = int(np.argmin(sims_to_M))
                new_medoids.append(far_idx)
                changed = True
                continue
            A = X[members]              # (m, d)
            # medoid minimizes sum(1 - a·x) = m - sum(a·x) -> maximize sum of sims
            G = A @ A.T                 # (m, m)
            best_local = int(members[np.argmax(G.sum(axis=1))])
            new_medoids.append(best_local)
            if best_local != medoids[c]:
                changed = True

        medoids = new_medoids
        if not changed:
            break

    return medoids

# ----------------- FULL-data evaluator (no UDFs) -----------------

def _dot_with_medoid_expr(med: np.ndarray) -> F.Column:
    """
    Build a Column computing dot(_arr, med) using ONLY SQL higher-order funcs:
      aggregate(zip_with(_arr, array(...), (a,b)->a*b), 0D, (acc,x)->acc+x)
    """
    arr_str = ",".join(f"CAST({float(v)} AS DOUBLE)" for v in med.tolist())
    return F.expr(
        f"aggregate(zip_with(_arr, array({arr_str}), (a,b) -> a*b), 0D, (acc, x) -> acc + x)"
    )

def _eval_full_cost_sum_cosine_sql(df_features: DataFrame, medoid_vecs: np.ndarray) -> float:
    """
    Sum over rows: 1 - max_j dot(_arr, med_j), computed via SQL expressions (no Python UDFs).
    df_features schema: [id_col, _arr(array<double>)] with L2-normalized arrays.
    """
    sim_cols = [_dot_with_medoid_expr(m) for m in medoid_vecs]
    max_sim = sim_cols[0] if len(sim_cols) == 1 else F.greatest(*sim_cols)
    dist_col = (F.lit(1.0) - max_sim).alias("_d")
    total = df_features.select(dist_col).agg(F.sum("_d").alias("_sum")).first()["_sum"]
    return float(total)

# ----------------- Main API -----------------

def method3_kmedoids_clara(
    df: DataFrame, *,
    id_col: str,
    numeric_cols: List[str],
    t_keep: int,
    sample_size: int = 20000,
    restarts: int = 8,
    pca_k: int = 0,
    seed: int = 42,
    cast_features_to_double: bool = True,
) -> List:
    """
    Return the IDs (id_col) of the t_keep medoids (R') using CLARA.
    - numeric_cols should EXCLUDE id_col.
    - Integer features are cast to DOUBLE (if cast_features_to_double=True), then vectorized,
      optional PCA (guarded), L2-normalized. PAM runs on a sample; best medoids evaluated on full data.

    Parameters
    ----------
    df : Spark DataFrame
    id_col : unique identifier column name (e.g., "unique_ID" or "Unique_ID")
    numeric_cols : list of feature column names (exclude id_col)
    t_keep : number of tuples to keep (k)
    sample_size : sample size per CLARA restart (20k–50k for ~1M rows)
    restarts : number of CLARA restarts (5–10 typical)
    pca_k : if >0, apply PCA to this dimension before normalization (auto-capped to #features)
    seed : RNG seed
    cast_features_to_double : cast features to DoubleType before VectorAssembler

    Returns
    -------
    List of id_col values (length t_keep)
    """
    assert t_keep > 0, "t_keep must be positive"
    if id_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != id_col]

    # Optional explicit cast of integer features to DOUBLE
    if cast_features_to_double:
        for c in numeric_cols:
            df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Vectorize
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="_raw_vec")
    base = assembler.transform(df).select(id_col, "_raw_vec").dropna(subset=["_raw_vec"])

    # Auto-cap PCA k
    if pca_k and pca_k > 0:
        num_feats = len(numeric_cols)
        if pca_k > num_feats:
            pca_k = num_feats

    # Optional PCA
    if pca_k and pca_k > 0:
        pca = MLPCA(k=pca_k, inputCol="_raw_vec", outputCol="_feat_vec")
        model = pca.fit(base)
        vdf = model.transform(base).select(id_col, "_feat_vec")
    else:
        vdf = base.withColumn("_feat_vec", F.col("_raw_vec")).select(id_col, "_feat_vec")

    # L2 normalize (Spark ML) and convert to array<double> without UDFs
    norm = Normalizer(inputCol="_feat_vec", outputCol="_norm", p=2.0)
    fdf = norm.transform(vdf) \
              .select(id_col, vector_to_array(F.col("_norm")).alias("_arr")) \
              .cache()

    n = fdf.count()
    if n == 0:
        return []
    if t_keep > n:
        raise ValueError(f"t_keep ({t_keep}) must be <= number of rows ({n}).")

    # CLARA sampling fraction
    frac = min(1.0, float(sample_size) / float(max(1, n)))

    best_cost = float("inf")
    best_ids: List = []

    for r in range(restarts):
        # Sample & collect (driver-side PAM on a small sample)
        samp = fdf.sample(False, frac, seed=seed + r)
        rows = samp.collect()
        if not rows:
            continue

        X = np.stack([np.asarray(row["_arr"], float) for row in rows], axis=0)
        ids = [row[id_col] for row in rows]

        # PAM on the sample
        med_idx = _pam_on_sample(X, t_keep, max_iter=50, seed=seed + r)
        med_vecs = X[med_idx]                 # (t_keep, d)
        med_ids  = [ids[i] for i in med_idx]

        # Evaluate on FULL data (pure SQL exprs)
        cost = _eval_full_cost_sum_cosine_sql(fdf, med_vecs)

        if cost < best_cost:
            best_cost = cost
            best_ids = med_ids

    return best_ids
