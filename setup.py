from setuptools import setup

setup(name='NumericalLuminosityInversion',
      version='1.0.0',
      description='Numerical inversion of emittance in Luminosity model',
      url='https://github.com/mattrufolo/Num_lumi_inversion/tree/main',
      author='Matteo Rufolo et al.',
      packages=['inversion_fun','inv_gauss_tree_maker'],
      zip_safe=False)