from py2neo import Graph
import pandas as pd


df = pd.read_csv("data/lnctard2.0.txt", delimiter='\t', encoding="latin-1", dtype="string")
df = df[["Regulator", "SearchregulatoryMechanism", "Target"]]
df = df.drop_duplicates().reset_index(drop=True)
# df = df[df["Regulator"].isin(df["Regulator"].value_counts().loc[lambda x: x > 1].index)].drop_duplicates().reset_index(drop=True)

graph = Graph("bolt://localhost:7687", auth=("neo4j", "123qweasd"))

for _, row in df.iterrows():
    h, r, t = row["Regulator"], row["SearchregulatoryMechanism"], row["Target"]
    query = f"MERGE (h:RNA{{name:'{h}'}})\n" \
            f"MERGE (t:RNA{{name:'{t}'}})\n" \
            f"MERGE (h)-[r:Regulatory{{name:'{r}'}}]->(t)"
    graph.run(query)

### docker command to run the container
# docker run --name neo4j -p7474:7474 -p7687:7687 --env NEO4J_AUTH=neo4j/123qweasd neo4j:latest

### run the cypher query in neo4j interface
# show the graph: MATCH (n) RETURN n
# delete the whole graph: MATCH (n) DETACH DELETE n

