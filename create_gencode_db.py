#!/usr/bin/env python3
import gffutils
import os

db_path = "data/gencode.v43.annotation.db"
if not os.path.exists(db_path):
  print("database does not exist, creating new one.")
  db = gffutils.create_db(
    "data/gencode.v43.annotation.gff3",
    dbfn=db_path,
    merge_strategy='create_unique', # TODO: we may want to do merge here
    verbose=True,
  )
else:
  print("loading existing database")
  db = gffutils.FeatureDB(db_path)
