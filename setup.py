
from setuptools import setup

setup(name='dg_python',
	version='1.0',
	description = 'Python implementation for simulating and fitting Dichotomous Gaussian models',
	url = 'https://github.com/mackelab/dg_python/tree/master/',	
	author = 'Poornima Ramesh',
	packages = ['dg_python'],
	install_requires=['numpy', 'scipy', 'matplotlib'],
	zip_safe = False)
