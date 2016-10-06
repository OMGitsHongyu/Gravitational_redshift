#Gravitational redshift

This is basically an analysis toolbox for Kaiser effects on Gadget output snapshot files.

## Installation

If your python with distutils is installed and properly configured, you can simply do

```
$ git clone https://github.com/OMGitsHongyu/Gravitational_redshift.git
$ cd Gravitational_redshift/analysis`
$ python setup.py build_ext --inplace
```

## Usage

`Gadget2Snapshot.py` is the io for gadget snapshot binary files.
`./subhalo/readgadgetmpi.py` is the input for gadget snapshot binary files in a parallel fashion.
`./analysis/calculator.pyx` is the analysis toolbox of density profiles.
`./corr2d/corr2d_kaiser.cpp` is the correlation function based on Kaiser effects (Redshift space distortion).
`./visualization/visual.py` is the visualization tool.

## Examples

See `./examples`
