# # # ### --------------------------------------------------------
# # # # # ------>>>>Python modules and packages 
# You are familiar with Python 
# scripts that can be executed 
# by running python <script-
# file>.py. Each of these files 
# has one or more import 
# statements to re-use Python 
# source code written by other 
# developers. These re-usable 
# Python source files are 
# referred to as modules. They 
# are .py Python source files, 
# like the scripts you have 
# been writing, that are stored 
# in a location where the 
# import statement can find 
# them. 

# There is no overtly 
# recognizable difference 
# between Python modules and 
# scripts, except that 
# developers of the former 
# write them with the intention 
# that their contents are 
# imported, rather than 
# executed by users. 

# In this chapter you'll learn 
# how to turn your Python 
# scripts into importable 
# modules, collect those 
# modules into packages and 
# create Conda packages that 
# can be easily installed by 
# other users. 

# Which of the following is 
# true about Python modules?
# R/ Python modules must be installed in order to use them.


# # # ### --------------------------------------------------------
# # # # # ------>>>> Importing a module
python pi.py compute_pi(1000)
# R/ ---> 3.14


# # # ### --------------------------------------------------------
# # # # # ------>>>> Modules and __name__
# The import statement was faster.


# # # ### --------------------------------------------------------
# # # # # ------>>>>Python package directory So 
# far you have seen that any 
# .py Python source file can be 
# used as a module to import 
# code. A Python package is a 
# collection of separate 
# modules collected under a 
# single name that share 
# metadata, such as 
# documentation, licensing, and 
# version numbering. 

# Here's an example directory 
# structure for my_package. 
# Note the repeated use of 
# my_package directories. The 
# individual modules are stored 
# in the inner my_package 
# directory. 

# my_package/ LICENSE README 
# setup.py my_package/ 
# __init__.py module1.py 
# module2.py ... Once this 
# package has been installed 
# you can import code from the 
# modules using the standard 
# import idioms. 

# Over the next several 
# exercises you're going to re-
# use much of the same code 
# that was present in the 
# mortgage_rates Anaconda 
# Project to prepare a Python 
# package that can be installed 
# and imported in new Python 
# projects. 

# The setup.py script file 
# coordinates installing the 
# package into your Python 
# distribution. 

# Why is it important to 
# package re-usable Python code?
# To keep a consistent version history of changes.
# To make module installation and management easier for others.
# To clearly define importable API components.
# R/ All of the above.



# # # ### --------------------------------------------------------
# # # # # # ------>>>>Importing a package A Python 
# package can have any number 
# of directories and module 
# source files. 

# The directory 
# mortgage_forecasts has been 
# prepared for you in your home 
# directory. It contains a 
# subdirectory of the same name 
# and two Python source code 
# files. 

# mortgage_forecasts/ 
# mortgage_forecasts/ models.py 
# utils.py models.py defines a 
# new class to fit and predict 
# 30-year mortgage rates in the 
# US. 

# utils.py defines functions to 
# read data and compute 
# statistical quantities. 

# Your working directory has 
# been set to 
# /home/repl/mortgage_forecasts. 
# Without changing directories, 
# which import statement will 
# provide access to the 
# MortgageRateModel class 
# defined in models.py? 

# You can use the terminal to 
# test import statements. A 
# conda environment has been 
# activated with the dependent 
# packages.
# R/ from mortgage_forecasts.models import MortgageRateModel



# # # ### --------------------------------------------------------
# # # # # ------>>>> The __init__.py file
echo "'''Predictive modeling of 30-year mortgage rates in the US.'''" > /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py
echo "from .models import MortgageRateModel" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py; echo "from .utils import read_data" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py


# # # ### --------------------------------------------------------
# # # # # ------>>>>Create the installer script
# Open any of the available editors to edit the file instead of using the following command
sed -i -e 's/name =.*/name = "mortgage_forecasts",/' /home/repl/mortgage_forecasts/setup.py; sed -i -e 's/description =.*/description = "30 year mortgage rate models",/' /home/repl/mortgage_forecasts/setup.py; sed -i -e 's/author =.*/author = "",/' /home/repl/mortgage_forecasts/setup.py
sed -i -e 's/setup(/setup( packages=find_packages(),/' /home/repl/mortgage_forecasts/setup.py

# # # # ### --------------------------------------------------------
# # # # # # ------>>>> Licensing Since our goal is 
# to share our code with others 
# we need to be aware of 
# copyright laws and the legal 
# rights we wish to retain 
# about how that software can 
# be used. Copyright 
# protections are guaranteed to 
# the person who owns the 
# software. When someone else 
# downloads and uses the 
# program we built we would not 
# want to transfer ownership to 
# them, thereby forfeiting our 
# rights. 

# Instead, we wish license 
# usage of the program under 
# certain restrictions. There 
# are many kinds of software 
# licenses and choosing a 
# license is beyond the scope 
# of this course. 

# For the mortgage_forecasts 
# package we want to share our 
# code with as few restrictions 
# as necessary. The only 
# important restriction is that 
# we would like to be 
# acknowledged as the original 
# author. For this we'll choose 
# the MIT license, which is 
# widely used in open-source 
# software packages. 

# The full text of the license 
# has been placed in the 
# mortgage_forecasts/LICENSE 
# file and it must remain in 
# the package directory for the 
# license to be valid and 
# enforceable. 

# Further, license="MIT" has 
# been added in setup.py. 

# which statement below is 
# INCORRECT?
# R/The MIT license does not allow commercial use.


# # # ### --------------------------------------------------------
# # # # # ------>>>> Version number
echo "__version__ = '0.1'" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py

# # # ### --------------------------------------------------------
# # # # # ------>>>>Install the package
python setup.py install
cd /home/repl/practice
# In the Python shell import MortgageRateModel and read_data from the mortgage_forecasts package and call:

# read_data() with the argument mortgage_rates.csv -> df
# MortgageRateModel() with the argument df -> model
# model.forecast() with the argument 'January 2019'
# You can continue to inspect the help() of the package, functions, and classes.

# Exit the interpreter when you're done.
