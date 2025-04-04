# SpatioTemporAl Mineral Prospectivity (STAMP) Modelling

[![DOI](https://zenodo.org/badge/587157997.svg)](https://zenodo.org/doi/10.5281/zenodo.10976576)

This repository contains the notebook and supplementary files required to reproduce the results presented in the paper titled _Machine learning-based spatiotemporal prospectivity modeling of porphyry systems in the New Guinea and Solomon Islands region_ by Farahbakhsh et al (2025). The notebook enables users to create a spatio-temporal prospectivity model for Porphyry Mineralisation in the New Guinea and Solomon Islands region.

## Input Files

The input files include plate motion models and oceanic grids provided by M&uuml;ller et al. (2016) and M&uuml;ller et al. (2019).

## Output Files

The final output is a set of GeoTIFF files that shows the probability of the targeted mineralisation in back-arc basins or a specific area at desired time steps.

# Dependencies

- pygplates
- gplately
- cartopy
- geopandas
- matplotlib
- netCDF4
- numpy
- pandas
- pulearn
- scipy
- shapely
- skimage
- sklearn
- skopt

# References

M&uuml;ller, R.D., Seton, M., Zahirovic, S., Williams, S.E., Matthews, K.J., Wright, N.M., Shephard, G.E., Maloney, K.T., Barnett-Moore, N., Hosseinpour, M. and Bower, D.J., 2016. Ocean basin evolution and global-scale plate reorganization events since Pangea breakup. Annual Review of Earth and Planetary Sciences, 44(1), pp.107-138.

M&uuml;ller, R.D., Zahirovic, S., Williams, S.E., Cannon, J., Seton, M., Bower, D.J., Tetley, M.G., Heine, C., Le Breton, E., Liu, S. and Russell, S.H., 2019. A global plate model including lithospheric deformation along major rifts and orogens since the Triassic. Tectonics, 38(6), pp.1884-1907.

# Cite

```bib
@article{Farahbakhsh2025,
  title = {Machine learning-based spatiotemporal prospectivity modeling of porphyry systems in the New Guinea and Solomon Islands region},
  author = {Farahbakhsh, Ehsan and Zahirovic, Sabin and McInnes, Brent I. A. and Polanco, Sara and Kohlmann, Fabian and Seton, Maria and M{\"u}ller, R. Dietmar},
  year = {2025},
  journal = {Tectonics},
  volume = {44},
  number = {3},
  pages = {e2024TC008362},
  doi = {10.1029/2024TC008362},
}
```
