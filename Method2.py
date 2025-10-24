from popularity import compute_popularity_in_batches


def method2 (df, queries, T, feature_cols,  id_col="Unique_ID"):
# T is how any tuples to keep
       pop_df = compute_popularity_in_batches (df, queries, id_col)
# calculate similarity matrix
       sim_df = compute_similarity (df, feature_cols, id_col)

#start with empty selection
       all_ids = [r[id_col]for r in df.select(id_col).collect()]
       selected_ids = []
#pick exactly T tuples
       for _ in range (T):
           best_candidate = None
           best_importance = -1

           for candidate_id in all_ids:
               if candidate_id in selected_ids:
                      continue

               candidate_set = selected_ids + [candidate_id]
               set_importance = calculate_set_importance_m2(candidate_set, pop_df, sim_df, id_col)


               if set_importance > best_importance:
                    best_importance = set_importance
                    best_candidate = candidate_id

           selected_ids.append(best_candidate)


       return df.filter(F.col(id_col).isin(selected_ids))

def calculate_set_importance_m2(selected_ids, pop_df, sim_df, id_col):
        if not selected_ids:
            return 0

        k = len(selected_ids)

        numerator_sum = 0

        # for each tuple in candidate set get popularity
        for i, t1 in enumerate(selected_ids):
            pop_t1 = pop_df.filter(F.col("id") == t1).select("pop").first()[0]

# how different it is from others in the set
            dissimilarity_sum = 0
            for j, t2 in enumerate (selected_ids):
                if i != j:
                    sim = sim_df.filter(
                        (F.col(f"{id_col}_1") == t1) & (F.col(f"{id_col}_2") == t2)
                    ).select("similarity").first()[0]
                    dissimilarity_sum += (1 - sim)

            avg_dissimilarity = dissimilarity_sum / (k-1) if k > 1 else 1
            numerator_sum += pop_t1 * avg_dissimilarity

        numerator = numerator_sum / k

        return numerator


