import os

import numpy as np
import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples.leakage import Sealant
from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline
from optuna.samplers import GridSampler

from matplotlib import pyplot as plt

print(pykeen.env())
df = pd.read_csv("data/lnctard2.0.txt", sep="\t", encoding="latin-1", dtype="string")
df = df[["Regulator", "SearchregulatoryMechanism", "Target"]]
df = df[df["Regulator"].isin(df["Regulator"].value_counts().loc[lambda x: x > 1].index)].drop_duplicates().reset_index(drop=True).to_numpy()
tf = TriplesFactory.from_labeled_triples(df)
training, valid, testing = tf.split([0.8, 0.1, 0.1], random_state=1234)

res = pd.DataFrame(
    columns=[
        'model', 'num_negs', 'batch_size', 'learning_rate', 'num_epochs',
        "hits_at_1", "hits_at_3", "hits_at_10", "arithmetic_mean_rank", "inverse_harmonic_mean_rank"
    ]
)

for model in ["TransE", "RotatE"]:
    for num_negs_per_pos in [16, 128]:
        for batch_size in [128, 256]:
            for lr, num_epochs in [(0.001, 200), (0.0001, 300)]:
                transe = pipeline(
                    training=training,
                    validation=valid,
                    testing=testing,
                    model=model,
                    model_kwargs=dict(embedding_dim=512),
                    training_kwargs=dict(use_tqdm_batch=False, num_epochs=num_epochs, batch_size=batch_size),
                    evaluation_kwargs=dict(use_tqdm=False),
                    optimizer_kwargs=dict(lr=lr),
                    negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
                    random_seed=1,
                    device="cuda",
                )
                metric = transe.metric_results.to_df()
                new_record = {
                    'model': model,
                    'num_negs': num_negs_per_pos,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'num_epochs': num_epochs,
                    "hits_at_1": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_1"), "Value"].values[0],
                    "hits_at_3": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_3"), "Value"].values[0],
                    "hits_at_10": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_10"), "Value"].values[0],
                    "arithmetic_mean_rank": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "arithmetic_mean_rank"), "Value"].values[0],
                    "inverse_arithmetic_mean_rank": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "inverse_arithmetic_mean_rank"), "Value"].values[0]
                }
                print(new_record)
                res = res.append(new_record, ignore_index=True)

res.to_csv("./pykeen_mrr_res.csv", sep="\t", index=False)