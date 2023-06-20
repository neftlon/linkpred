FROM ubuntu:20.04

RUN apt update && apt install -y git python3 python3-pip

# install Python dependencies
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
RUN pip install decorator numpy matplotlib tqdm networkx ninja jinja2 rdkit-pypi
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# install torchdrug
RUN git clone https://github.com/DeepGraphLearning/torchdrug.git && \
  cd torchdrug/ && pip install -r requirements.txt && python setup.py install
  
# install some more dependencies  
RUN pip install ogb easydict pyyaml
