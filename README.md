![NeSSI logo](https://raw.githubusercontent.com/wiki/PageotD/nessi/images/nessi.png)
![Powered by PYTHON](https://www.python.org/static/community_logos/python-powered-w-100x40.png)
![LGPL3 logo](https://www.gnu.org/graphics/lgplv3-88x31.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1346007.svg)](https://doi.org/10.5281/zenodo.1346007)

NeSSI (Near-Surface Seismic Imaging) aims to provide python modules for the rapid development of seismic inversion codes based on the particle swarm optimization method.

NeSSI is an open-source project licensed under the [LGPLv3](http://www.gnu.org/licenses/lgpl-3.0-standalone.html).

## Documentation

The [NeSSI documentation](https://pageotd.github.io/nessi/) provides:
* Getting started informations
* Tutorials
* NeSSI API

The [NeSSI wiki]() provides informations about:
* [How to contribute](https://github.com/PageotD/nessi/blob/master/CONTRIBUTE.md)
* [How to pull issues](https://github.com/PageotD/nessi/blob/master/CONTRIBUTE.md)

Some examples in the form of [jupyter notebooks](http://jupyter.org/) are availables in the [nessi.materials](https://github.com/PageotD/nessi.materials) repository.

## References

* An example of application using the 2D PSv modeling engine and PSO to invert Rayleigh dispersion diagrams: [Pageot et al., Alternative method for surface wave inversion, _Congrès Français d'Acoustique, Le Havre, 2018_](https://www.researchgate.net/publication/324889746_Methode_alternative_d'inversion_des_ondes_de_surface)

## Install
First, install __git__ on your computer. For exemple on Ubuntu:

`sudo apt-get install git`

or use the `package manager`.

You also need __gcc/gfortran__, __numpy__, __scipy__ and __matplotlib__.

Download __NeSSI__ with:

`git clone https://github.com/PageotD/nessi.git`

In the nessi folder, check for update:
`git pull`

Then, in a terminal, go to the nessi/nessi folder and type:

`make`

Finally, you have to add the path of the NeSSI package to `PYTHONPATH`. In your `~/.bashrc` file, add:

`export PYTHONPATH=$PYTHONPATH:/path/to/the/nessi/package`
