## experiment ideas

### evaluate data:
* edge-fixed new data with Complete() transformation
* same data with AddEdges(np.inf)
  
one run each with 300 epochs  
if they work: merge into master


### data
* compare manually constructed features from Tencent with raw data
* implicit vs explicit hydrogen atoms

### new architectures
* adapt DimeNet pytorch implementation

### modifications of Tencent MPNN
* graph state  
  check: mpnn_root.ipynb
* Tencent MPNN with edge updates
* set2set vs simple note summation

### other
* rotate and translate molecules and measure the variance of predictions of different architectures