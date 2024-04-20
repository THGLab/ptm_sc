PTM sidechain analysis & packing
=================================

Supports sidechain torsion extraction, visualization and kernel regression for phosphoralytion, acetylation & methylation.
The generated libraries are now part of MCSCE (https://github.com/THGLab/MCSCE).

Environment requirements
=======================
- numpy
- pandas
- scipy
- biopython
- matplotlib (for visualization)

To create a dedicated conda environment:
```
conda create -n ptm python=3.9
conda activate ptm
conda install ipykernel
conda install numpy pandas matplotlib
pip install scipy==1.9.3 biopython==1.79
```

Running jupyter notebooks:
```
python -m ipykernel install --user --name ptm
```

Reference
=========

