# # # ### --------------------------------------------------------
# # # # # ------>>>>Conda Packages In the last 
# chapter you created a Python 
# package and successfully 
# installed it. However, you 
# needed to have 1) downloaded 
# the source code, and 2) 
# created a Conda Environment 
# with the dependent packages. 

# Further, when using 
# setuptools to install 
# packages there are no 
# uninstall or update commands. 
# That means you would have to 
# manually remove the installed 
# files if you want to install 
# a newer version of the 
# package. As you saw in the 
# Conda Essentials course, 
# Conda packages solve each of 
# these issues, but you might 
# use pip and virtualenv as 
# well. 

# In this chapter you'll create 
# a Conda Recipe for the 
# mortgage_forecasts package to 
# define the dependent Conda 
# packages. The Conda recipe is 
# specified in a file called 
# meta.yaml. 

# You'll then build the package 
# archive and upload it to 
# Anaconda Cloud. Further, 
# Conda packages are not 
# limited to Python packages. A 
# package written in any 
# programming language, or a 
# collection of files, can 
# become a Conda package. 

# Which statement below is 
# INCORRECT?
# Having a setup.py file is enough to build a Conda package.

# # # ### --------------------------------------------------------
# # # # # ------>>>> Install Conda Build
conda install conda-build -y

# # # # ### --------------------------------------------------------
# # # # # # ------>>>> The Conda recipe meta.yaml
# The mortgage_forecasts 
# package you wrote in the last 
# chapter has been provided in 
# your home directory along 
# with a template meta.yaml 
# Conda recipe. There are 5 
# sections of this file: 

# package defines the package 
# name and version source 
# provides the relative path (
# or Github repository) to the 
# package source files build 
# defines the command to 
# install the package 
# requirements specify the 
# conda packages required to 
# build and run the package 
# about provides other 
# important metadata like the 
# license and description 
# Inspect the meta.yaml file 
# with nano, vim, emacs, cat, 
# less or more. 

# You'll see at the top of the 
# file {% set setup_py = 
# load_setup_py_data() %}. When 
# the package is built metadata 
# like the version number and 
# license will be read directly 
# from the setup.py file in the 
# source path. 

# Read the meta.yaml 
# documentation for more 
# details. 

# Why are there build: and run: 
# sections in requirements:?
# R/Build packages are those required by setup.py.


# # # ### --------------------------------------------------------
# # # # # # ------>>>>Conda package dependencies
# In the meta.yaml file dependent packages and versions are defined using 
# comparison operators, such as <, <=, >, >=. Multiple version conditions
#  are separated by commas. The glob * means that the preceding characters must match exactly.

# Here's an example for a new package called my_package.

# requirements:
#     run:
#         - python
#         - scipy
#         - numpy 1.11*
#         - matplotlib >=1.5, <2.0
# Which of the following statements below is INCORRECT?
# R/NumPy version 1.13 is compatible with my_package.


# # # ### --------------------------------------------------------
# # # # # ------>>>>Dependent package versions
sed -i -e 's/^    run:/    run:\n         - python >=2.7\n        - pandas >=0.20/n        - statsmodels\n        - scipy\n' /home/repl/mortgage_forecasts/meta.yaml
sed -i -e 's/^    build:/    build:\n        - python\n        - setuptools/' /home/repl/mortgage_forecasts/meta.yaml


# # # ### --------------------------------------------------------
# # # # # ------>>>>Build the Conda Package
conda build mortgage_forecasts
conda search --use-local --info mortgage_forecasts


# # # ### --------------------------------------------------------
# # # # # ------>>>Install the conda package
conda create -n conda-models python=3
conda activate models 
conda install mortgage_forecasts pandas=0.19 --use-local
conda install mortgage_forecasts pandas=0.22 --use-local


# # # ### --------------------------------------------------------
# # # # # ------>>>Python versions and architectures
# Edit meta.yaml to add the following line to the build: tag and before the script: tag.

#     noarch: python
#     number: 1
conda build .


# # # ### --------------------------------------------------------
# # # # # ------>>>Upload the package
anaconda upload noarch/mortgage_forecasts-0.1-py_1.tar.bz2 --username datacamp-student --password datacamp1
