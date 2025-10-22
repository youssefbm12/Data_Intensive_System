import random
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
import random


from pyspark.sql import functions as F
import random
import numpy as np


def get_column_stats(relational_db):
    """
    Compute min, max, mean, stddev for numeric (int/float) columns.
    """
    col_stats = {}
    for table_name, sdf in relational_db.items():
        numeric_cols = [
            f.name for f in sdf.schema.fields
            if f.dataType.simpleString() in ('int', 'bigint', 'float', 'double')
        ]
        if not numeric_cols:
            continue

        agg_exprs = []
        for col in numeric_cols:
            agg_exprs.extend([
                F.min(col).alias(f"{col}_min"),
                F.max(col).alias(f"{col}_max"),
                F.mean(col).alias(f"{col}_mean"),
                F.stddev(col).alias(f"{col}_std")
            ])
        stats = sdf.select(agg_exprs).collect()[0]

        col_stats[table_name] = {
            col: {
                "min": stats[f"{col}_min"],
                "max": stats[f"{col}_max"],
                "mean": stats[f"{col}_mean"],
                "std": stats[f"{col}_std"]
            }
            for col in numeric_cols
        }
    return col_stats


def generate_random_queries(df, table_name, col_stats, n_queries=5, max_conditions=2):
    """
    Generate realistic random SQL queries based on column distributions (int values),
    always including 'patient_BSN' in the SELECT clause.
    """
    columns = df.columns
    queries = []
    table_info = col_stats.get(table_name, {})

    for _ in range(n_queries):
        # --- SELECT clause ---
        n_select = random.randint(1, min(3, len(columns)))
        selected_cols = random.sample(columns, n_select)

        # Always include 'patient_BSN'
        if "Unique_ID" not in selected_cols and "Unique_ID" in columns:
            selected_cols.insert(0, "Unique_ID")

        select_clause = ", ".join(selected_cols)

        # --- WHERE clause ---
        n_conditions = random.randint(1, max_conditions)
        conditions = []
        numeric_cols = list(table_info.keys())

        for _ in range(n_conditions):
            if not numeric_cols:
                break
            col = random.choice(numeric_cols)
            stats = table_info[col]

            if None in stats.values():
                continue

            # Sample integer value from normal distribution
            sampled_val = np.random.normal(stats["mean"], stats["std"])
            sampled_val = int(round(max(min(sampled_val, stats["max"]), stats["min"])))

            conditions.append(f"{col} = {sampled_val}")

        where_clause = ""
        if conditions:
            connector = random.choice(["AND", "OR"])
            where_clause = " WHERE " + f" {connector} ".join(conditions)

        # --- Final query ---
        query = f"SELECT {select_clause} FROM {table_name}{where_clause};"
        queries.append(query)

    return queries


def generate_query_dicts(df, table_name, col_stats, n_queries=5, max_conditions=2):
    """
    Generate a list of query dicts safe for compute_popularity.
    Ensures at least one condition per query and all columns exist.
    """
    queries = []
    table_info = col_stats.get(table_name, {})
    numeric_cols = list(table_info.keys())

    if not numeric_cols:
        return [{} for _ in range(n_queries)]  # fallback: empty queries

    for _ in range(n_queries):
        n_conditions = random.randint(1, min(max_conditions, len(numeric_cols)))
        query_dict = {}

        # Pick unique columns to avoid duplicates
        cols = random.sample(numeric_cols, n_conditions)

        for col in cols:
            stats = table_info[col]
            if None in stats.values():
                continue
            sampled_val = np.random.normal(stats["mean"], stats["std"])
            sampled_val = int(round(max(min(sampled_val, stats["max"]), stats["min"])))
            query_dict[col] = sampled_val

        if not query_dict:
            # fallback: pick a random column with min value
            col = random.choice(numeric_cols)
            query_dict[col] = int(table_info[col]["min"])

        queries.append(query_dict)

    return queries
