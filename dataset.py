import pandas as pd
import networkx as nx
import os
import torch
from torch.utils import data as torch_data
from torchdrug import data as torchdrug_data
from torch_geometric.data import InMemoryDataset

# this code is adjusted from https://github.com/DeepGraphLearning/torchdrug/blob/a959f68f0c19f368be9e380f5a587de6970b3c67/torchdrug/datasets/fb15k.py#L10
class TdLncTarD(torchdrug_data.KnowledgeGraphDataset):
  """
  torchdrug representation of LncTarD dataset.
  """
  
  def __init__(self, verbose=1):
    splits = ["train", "val", "test"]
    # TODO(johannes): find out whether this also removes the .tsv file header!
    self.load_tsvs([f"data/{s}.tsv" for s in splits], verbose=verbose)
  
  def split(self):
    offset = 0
    splits = []
    for num_sample in self.num_samples:
      split = torch_data.Subset(self, range(offset, offset + num_sample))
      splits.append(split)
      offset += num_sample
    return splits

class PygLncTarD(InMemoryDataset):
  """
  pytorch geometric compatible version of LncTarD dataset.
  
  NB: The class is not (yet) capable of downloading the dataset itself.
    Instead, it expects the dataset to be pre-split and put inside the 
    `data/` subdirectory such that it can be loaded with `load_split`.
  """
  
  def __init__(self, split_name, transform=None, device=None, verbose=1):
    super().__init__(".", transform)
    
    # load data
    self.split_name = split_name
    self.df = load_split(split_name)
    # use full_df to obtain mappings, to ensure that all possible heads, 
    # targets, and relations are available.
    self.full_df = load_split("full")
    
    # load mappings for one hot encoding from original dataset, such that
    # all possible entities (=heads or tails) and relations are available.
    # NB: pyg does not distinguish between head and tail entities, therefore
    # they are concatenated to a large list of entities here
    full_df = load_split("full")
    entities = pd.concat([full_df["head"], full_df["tail"]], ignore_index=True).drop_duplicates(ignore_index=True)
    self.ent2idx = pd.get_dummies(entities,dtype=float).idxmax()
    self.rel2idx = pd.get_dummies(full_df["relation"].drop_duplicates(ignore_index=True),dtype=float).idxmax()
    self.idx2ent, self.idx2rel = self.ent2idx.T, self.rel2idx.T
    
    # NB: this is not a very efficient way of obtaining the dataset, but it
    # is sufficient for a small dataset like LncTarD.
    self.tuples = torch.zeros((len(self.df), 3), dtype=torch.long, device=device)
    num_failed = 0
    for tup in self.df.itertuples(index=True):
      # TODO: figure out how we can ignore the weird string encodings, also we
      #   need to clamp tuples to the appropriate size when doing this!!!
      try:
        h, r, t = self.ent2idx[tup.head], self.rel2idx[tup.relation], self.ent2idx[tup.tail]
        assert tup.Index < len(self.tuples), "tuple index out of range"
        self.tuples[tup.Index] = torch.tensor([h,r,t])
      except KeyError:
        num_failed += 1
    if verbose:
      print("cannot load %d/%d tuple(s) from %s because of weird string encoding" % 
            (num_failed, num_failed + len(self.tuples), split_name))

  def len(self):
    return len(self.tuples)
  
  def get(self, idx):
    return self.tuples[idx]
  
  @property
  def num_nodes(self):
    """Return the number of node types available in this dataset."""
    # NB: in https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/base.html#KGEModel
    # it looks like that this should return the # of available node TYPES.
    return len(self.ent2idx)
  
  @property
  def num_edge_types(self):
    """Return the number of edge types available in this dataset."""
    return len(self.rel2idx)
  
  @property
  def edge_index(self):
    """
    Return the edge indices of this dataset, as a tuple of heads and tails.
    """
    return self.tuples[:,[0,2]].T
  
  @property
  def edge_type(self):
    """Return the edge types as a tensor."""
    return self.tuples[:,1]

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
