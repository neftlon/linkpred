import networkx as nx
import pandas as pd
import random
import pytest
from linkpred.metrics import hits_at_k, mean_rank, mean_reciprocal_rank
from numpy import isclose

@pytest.fixture
def G():
    edges = [
    ('A', 'r1', 'B'),
    ]
    df = pd.DataFrame(edges, columns=["head", "relation", "tail"])
    G = nx.from_pandas_edgelist(df, source="head", target="tail", edge_attr="relation")
    return G

@pytest.fixture
def score_list(G):
    random.seed(10)
    r1_score_list = []
    for n1 in G.nodes():
        for n2 in G.nodes():
            r1_score_list.append((n1, n2, random.random()))
    return r1_score_list

def test_score_list(score_list):
    true_sorted_score_list = [
        ('B', 'A', 0.5780913011344704),
        ('A', 'A', 0.5714025946899135),
        ('A', 'B', 0.4288890546751146),
        ('B', 'B', 0.20609823213950174)
    ]
    sorted_list = sorted(score_list, key=lambda x: x[2], reverse=True)
    
    for i in range(len(true_sorted_score_list)):
        assert sorted_list[i][0] == true_sorted_score_list[i][0]
        assert sorted_list[i][1] == true_sorted_score_list[i][1]
        assert isclose(sorted_list[i][2], true_sorted_score_list[i][2])

def test_hits_at_1(score_list, G):
    assert isclose(hits_at_k(score_list, G.edges(), k=1), (0, 0)).all()

def test_hits_at_3(score_list, G):
    assert isclose(hits_at_k(score_list, G.edges(), k=3), (1/3, 1)).all()

def test_hits_at_4(score_list, G):
    assert isclose(hits_at_k(score_list, G.edges(), k=4), (1/4, 1)).all()

def test_mean_rank(score_list, G):
    assert isclose(mean_rank(score_list, G.edges()), 3.0)

def test_mean_reciprocal_rank(score_list, G):
    assert isclose(mean_reciprocal_rank(score_list, G.edges()), 1/3)