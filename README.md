# GNN Collective Cell Dynamics

This repository contains example codes of our recent work:

"Learning Dynamics from Multicellular Graphs with Deep Neural Networks", by H Yang, F Meyer, S Huang, L Yang, C Lungu, M A Olayioye, M Buehler and M Guo.
[Link to the arXiv paper](https://arxiv.org/abs/2401.12196)

Here we provide an example code (GraphNetCellMain.py) to train a GNN model consisting of PNA layers (GraphNetCellModel.py) to approximate the relation between a static snapshot of cell monolayer and the cell mobility.

The preprocessing code (GraphNetCellPreProcess.py) is also provided here. From a 2-D point cloud (e.g. cell coordinates), it generates Delaunay tessellation as graph edges, and calculates area and perimeter of the Voronoi tessellation as node features.
