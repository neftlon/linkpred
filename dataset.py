import pandas as pd
import networkx as nx
import os

def load_lnctard(filename: str = "data/lnctard2.0.txt", cols: list[str] = None):
  """
  Load the entire LncTarD dataset and select only wanted `cols`.
  """
  if cols is None:
    cols = ["Regulator", "Target", "SearchregulatoryMechanism", 
            "RegulatorType", "TargetType"]
  lnctard = pd.read_csv(filename, sep="\t", encoding="latin-1")
  # TODO: decide whether we want to drop duplicates from the dataset here
  return lnctard[cols].drop_duplicates()

def load_split(name: str):
  """
  Load a dataset split from the `data/` directory.
  
  Args:
    name: Which split to take. Can be either `"train"`, `"val"`, `"test"`, or
      `"full"`. The first three load the associated split if available, the
      latter one loads the entire dataset.
  """
  filename = f"data/{'dataset' if name == 'full' else name}.tsv"
  assert os.path.exists(filename), (
    "cannot load split for %s, file not found" % name
  )
  return pd.read_csv(filename, sep="\t", encoding="latin-1")

def df2nx(
  df, head="head", tail="tail", relation="relation", cc_mode: str = "largest"
):
  """
  Convert columns of a pandas DataFrame to a nx.DiGraph.
  
  Args:
    df: DataFrame with at least columns `head`, `tail`, and `relation`.
    head: Field of `df` containing source nodes of relation.
    tail: Field of `df` containing target nodes of relation.
    relation: Field of `df` indicating relation type.
    cc_mode: Connection component mode: specify return type of value. Can be
      either `"largest"`, `"all"`, or `"G"`. `"largest"` mode returns only the
      subgraph formed by the largest connected component. `"all"` returns a 
      tuple containing the graph and a list of sorted (by length) node sets, 
      belonging to each connection component. `"G"` returns the graph as is.
  """
  G = nx.from_pandas_edgelist(
    df, source=head, target=tail, edge_attr=relation, create_using=nx.DiGraph(),
  )
  if cc_mode == "G":
    return G
  conn_comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
  if cc_mode == "largest":
    assert len(conn_comps) > 0, (
      "provided graph does not contain weakly connected components"
    )
    return G.subgraph(conn_comps[0])
  elif cc_mode == "all":
    return G, conn_comps
  else:
    raise ValueError("invalid `cc_mode` selected.")
  
  
