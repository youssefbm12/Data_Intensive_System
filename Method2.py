

from popularity import compute_popularity_in_batches
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
import numpy as np



def compute_similarity(df, feature_cols, id_col="Unique_ID"):
    """Compute pairwise cosine similarity between all tuples"""
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vector = assembler.transform(df)

    vectors_rdd = df_vector.select(id_col, "features").rdd
    vectors_dict = vectors_rdd.collectAsMap()

    all_ids = list(vectors_dict.keys())
    pairs = []

    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            id1, id2 = all_ids[i], all_ids[j]
            vec1 = vectors_dict[id1].toArray()
            vec2 = vectors_dict[id2].toArray()

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            similarity = float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
            pairs.append((id1, id2, similarity))

    sim_df = spark.createDataFrame(pairs, [f"{id_col}_1", f"{id_col}_2", "similarity"])

    self_sim = spark.createDataFrame(
        [(id, id, 1.0) for id in all_ids],
        [f"{id_col}_1", f"{id_col}_2", "similarity"]
    )

    symmetric = sim_df.select(
        F.col(f"{id_col}_2").alias(f"{id_col}_1"),
        F.col(f"{id_col}_1").alias(f"{id_col}_2"),
        F.col("similarity")
    )

    return sim_df.union(symmetric).union(self_sim)

def get_similarity(sim_df, id1, id2, id_col="Unique_ID"):
    #similarity bet 2 tuples
    result = sim_df.filter(
        (F.col(f"{id_col}_1") == id1) & (F.col(f"{id_col}_2") == id2)
    ).select("similarity").first()
    return result[0] if result else 0.0


def method2(df, queries, T, feature_cols, id_col="Unique_ID"):
    pop_df = compute_popularity_in_batches(df, queries, id_col)
    sim_df = compute_similarity(df, feature_cols, id_col)

    pop_dict = {row["id"]: row["pop"] for row in pop_df.collect()}
    all_ids = [r[id_col] for r in df.select(id_col).collect()]
    selected_ids = []

    for _ in range(T):
        best_candidate = None
        best_importance = -1

        for candidate_id in all_ids:
            if candidate_id in selected_ids:
                continue

            candidate_set = selected_ids + [candidate_id]
            set_importance = calculate_set_importance_m2(candidate_set, pop_dict, sim_df, id_col)

            if set_importance > best_importance:
                best_importance = set_importance
                best_candidate = candidate_id

        selected_ids.append(best_candidate)

    return df.filter(F.col(id_col).isin(selected_ids))


def calculate_set_importance_m2(selected_ids, pop_dict, sim_df, id_col):
    if not selected_ids:
        return 0

    k = len(selected_ids)
    numerator_sum = 0

    for i, t1 in enumerate(selected_ids):
        pop_t1 = pop_dict.get(t1, 0)
        dissimilarity_sum = 0
        count = 0

        for j, t2 in enumerate(selected_ids):
            if i != j:
                sim = get_similarity(sim_df, t1, t2, id_col)
                dissimilarity_sum += (1 - sim)
                count += 1

        avg_dissimilarity = dissimilarity_sum / count if count > 0 else 1
        numerator_sum += pop_t1 * avg_dissimilarity

    return numerator_sum / k

