# 360MonoDepth

Based on [360MonoDepth](https://manurare.github.io/360monodepth/) by [Manuel Rey-Area](https://manurare.github.io/), [Mingze Yuan](https://yuanmingze.github.io/) and [Christian Richardt](https://richardt.name/)


## Setup

Tested with Python >= 3.8

1. Clone the repository

```bash
git clone git@github.com:patriciogonzalezvivo/360monodepth.git
cd 360monodepth
```

2. Install C++ dependencies for:

 * Ceres 2.0.0
 * Eigen 3.3.9
 * Glog 0.5.0
 * Gflags 2.2.2
 * GTest 1.10.0
 * OpenCV 4.2.0
 * Boost 1.75.0
 * pybind11 2.8.1

On linux:

```
sudo apt install libceres-dev libeigen3-dev libgoogle-glog-dev libgflags-dev libgtest-dev libopencv-dev libboost-all-dev python3-pybind11 pybind11-dev
```

3. We need to create a conda environment with python 3.8 and build the C++ targets

```
conda create -f environment.yml
conda activate 360monodepth
```

4. Build Python bindings

```
cd cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j8
```

5. Copy all the dependent DLL/so files to ```cpp/python/[dll,so]```. For example, in Linux ```cpp/python/so``` should contain the following dynamic libraries: ```libamd.so.2, libcholmod.so.3, libglog.so, libm.so.6, libsuitesparseconfig.so.5, libblas.so.3, libcolamd.so.2, libglog.so.0, libopencv_core.so.4.2, libtbb.so.2, libcamd.so.2, libcxsparse.so.3, libgomp.so.1, libopencv_imgproc.so.4.2, libccolamd.so.2, libgflags.so.2.2, liblapack.so.3, libquadmath.so.0, libceres.so.2, libgfortran.so.5, libmetis.so.5, libspqr.so.2```

```
cd cpp/python
python setup.py build
python setup.py bdist_wheel
pip install dist/instaOmniDepth-0.1.0-cp39-cp39-linux_x86_64.whl 
```

## Running code

```
python main.py -i data/0001.jpg -o data/0001 --grid_size 8x7 --persp_monodepth=midas3
```

## Citation

```
@inproceedings{reyarea2021360monodepth,
	title={{360MonoDepth}: High-Resolution 360{\deg} Monocular Depth Estimation},
	author={Manuel Rey-Area and Mingze Yuan and Christian Richardt},
	booktitle={CVPR},
	year={2022}}
```
