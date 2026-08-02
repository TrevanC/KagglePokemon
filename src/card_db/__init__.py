"""Static card database: CSV parsing and precomputed feature table.

Everything here runs *offline*. Design rule 2: never parse EN_Card_Data.csv
inside the live Kaggle turn loop — read data/processed/static_feature_table.pt.
"""
