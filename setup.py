from setuptools import setup, find_packages

with open("README.md", "r") as fr:
    long_description = fr.read()

setup(name='FerroSim',
      version='0.0.1',
      description='FerroSim 2D simulation',
      long_description = long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/ramav87/ferrosim',
      author='Rama Vasudevan',
      license='MIT',
      packages=find_packages(exclude='tests'),
      zip_safe=False)
