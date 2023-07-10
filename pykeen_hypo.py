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


transe_hypo = hpo_pipeline(
    training=training,
    validation=valid,
    testing=testing,
    device="cuda",
    model="TransE",
    n_trials=50,
    sampler=GridSampler,
    sampler_kwargs=dict(
        search_space={
            "model.embedding_dim": [512],
            "model.scoring_fct_norm": [2],
            "loss.margin": [1.0],
            "optimizer.lr": [1.0e-03, 1.0e-04],
            "negative_sampler.num_negs_per_pos": [16, 32, 64, 128],
            "training.num_epochs": [300],
            "training.batch_size": [32, 64, 128, 256, 512, 1024],
        },
    ),
)
