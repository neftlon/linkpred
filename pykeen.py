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
    model_kwargs=dict(embedding_dim=512),
    training_kwargs=dict(use_tqdm_batch=False, num_epochs=200),
    n_trials=50,
    training_kwargs_ranges=dict(
        batch_size=dict(type='categorical', choices=[32, 64, 128, 256, 512, 1024], log=True)
        # batch_size=dict(type='categorical', choices=[32, 64], log=True)
    ),
    evaluation_kwargs=dict(use_tqdm=False),
    negative_sampler_kwargs_ranges=dict(
        num_negs_per_pos=dict(type='categorical', choices=[16, 32, 64, 128], log=True)
        # num_negs_per_pos=dict(type='categorical', choices=[16, 32], log=True)
    ),
)
