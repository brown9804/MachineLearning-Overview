# ### --------------------------------------------------------
# # # ------>>>>What feature is NOT true of 
# Conda packages?
# R/Each package contains binary artifacts (executables) for all supported platforms.


# ### --------------------------------------------------------
# # # ------>>>> What version of conda do I have?
conda --version

# ### --------------------------------------------------------
# # # ------>>>>Install a conda package (I)
# Just as you can get help on conda as a whole, 
# you can get help on commands within it. You will 
# often use the command conda install; you can look
#  at the corresponding help documentation using the 
#  terminal window. That is, run conda install --help 
#  and read through the output.

# How is the positional argument package_spec defined in
#  the documentation for conda install?
# R/ Packages to install into the conda environment.


# ### --------------------------------------------------------
# # # ------>>>>Install a conda package (II) 
# Installing a package is 
# largely a matter of listing 
# the name(s) of packages to 
# install after the command 
# conda install. But there is 
# more to it behind the scenes. 
# The versions of packages to 
# install (along with all their 
# dependencies) must be 
# compatible with all versions 
# of other software currently 
# installed. Often this 
# "satisfiability" constraint 
# depends on choosing a package 
# version compatible with a 
# particular version of Python 
# that is installed. Conda is 
# special among "package 
# managers" in that it always 
# guarantees this consistency; 
# you will see the phrase 
# "Solving environment..." 
# during installation to 
# indicate this computation. 

# For example, you may simply 
# instruct conda to install foo-
# lib. The tool first 
# determines which operating 
# system you are running, and 
# then narrows the match to 
# candidates made for this 
# platform. Then, conda 
# determines the version of 
# Python on the system (say 
# 3.7), and chooses the package 
# version for -py37. But, 
# beyond those simple limits, 
# all dependencies are checked. 

# Suppose foo-lib is available 
# in versions 1.0, 1.1, 1.2, 
# 2.0, 2.1, 2.2, 2.3, 3.0, 3.1 (
# for your platform and Python 
# version). As a first goal, 
# conda attempts to choose the 
# latest version of foo-lib. 
# However, maybe foo-lib 
# depends on bar-lib, which 
# itself is available in 
# various versions (say 1 
# through 20 in its versioning 
# scheme). It might be that foo-
# lib 3.1 is compatible with 
# bar-lib versions 17, 18, and 
# 19; but blob-lib (which is 
# already installed) is 
# compatible only with versions 
# of bar-lib less than 17. 
# Therefore, conda would 
# examine the compatibility of 
# foo-lib 3.0 as a fallback. In 
# this hypothetical, foo-lib 
# 3.0 is compatible with bar-
# lib 16, so that version is 
# chosen (bar-lib is also 
# updated to the latest 
# compatible version 16 in the 
# same command if an earlier 
# version is currently 
# installed).
conda install cytoolz
y

# ### --------------------------------------------------------
# # # ------>>>>What is semantic versioning? 
# Most Conda packages use a 
# system called semantic 
# versioning to identify 
# distinct versions of a 
# software package 
# unambiguously. Version labels 
# are usually chosen by the 
# project authors, not 
# necessarily the same people 
# who bundle the project as a 
# Conda package. There is no 
# technical requirement that a 
# project author's version 
# label coincides with a Conda 
# package version label, but 
# the convention of doing so is 
# almost always followed. 
# Semantic version labels can 
# be compared lexicographically 
# and make it easy to determine 
# which of two versions is the 
# later version. 

# Under semantic versioning, 
# software is labeled with a 
# three-part version identifier 
# of the form 
# MAJOR.MINOR.PATCH; the label 
# components are non-negative 
# integers separated by 
# periods. Assuming all 
# software starts at version 
# 0.0.0, the MAJOR version 
# number is increased when 
# significant new functionality 
# is introduced (often with 
# corresponding API changes). 
# Increases in the MINOR 
# version number generally 
# reflect improvements (e.g., 
# new features) that avoid 
# backward-incompatible API 
# changes. For instance, adding 
# an optional argument to a 
# function API (in a way that 
# allows old code to run 
# unchanged) is a change worthy 
# of increasing the MINOR 
# version number. An increment 
# to the PATCH version number 
# is appropriate mostly for bug 
# fixes that preserve the same 
# MAJOR and MINOR revision 
# numbers. Software patches do 
# not typically introduce new 
# features or change APIs at 
# all (except sometimes to 
# address security issues). 

# Many command-line tools 
# display their version 
# identifier by running tool --
# version. This information can 
# sometimes be displayed or 
# documented in other ways. For 
# example, suppose on some 
# system, a certain version of 
# Python is installed, and you 
# inquire about it like this: 

# python -c "import sys; 
# sys.version" '1.0.1 (Mar 26 
# 2014)' Looking at the output 
# above, which statement below 
# accurately characterizes the 
# semantic versioning of this 
# installed Python?
# The MAJOR version is 1, the PATCH is 1

# ### --------------------------------------------------------
# # # ------>>>>Which package version is installed?
conda list # then search for request version
2.22.0


# ### --------------------------------------------------------
# # # ------>>>>Install a specific version of a package (I)
conda install attrs=17.3
y


# ### --------------------------------------------------------
# # # ------>>>>Install a specific version of a package (II)
# Most commonly, 
# you'll use prefix-notation to 
# specify the package version(
# s) to install. But conda 
# offers even more powerful 
# comparison operations to 
# narrow versions. For example, 
# if you wish to install either 
# bar-lib versions 1.0, 1.4 or 
# 1.4.1b2, but definitely not 
# version 1.1, 1.2 or 1.3, you 
# could use: 

# conda install 'bar-lib=1.0|
# 1.4*' This may seem odd, but 
# you might know, for example, 
# that a bug was introduced in 
# 1.1 that wasn't fixed until 
# 1.4. You would prefer the 1.4 
# series, but, if it is 
# incompatible with other 
# packages, you can settle for 
# 1.0. Notice we have used 
# single quotes around the 
# version expression in this 
# case because several of the 
# symbols in these more complex 
# patterns have special 
# meanings in terminal shells. 
# It is easiest just to quote 
# them. 

# With conda you can also use 
# inequality comparisons to 
# select candidate versions (
# still resolving dependency 
# consistency). Maybe the bug 
# above was fixed in 1.3.5, and 
# you would like either the 
# latest version available (
# perhaps even 1.5 or 2.0 have 
# come out), but still avoiding 
# versions 1.1 through 1.3.4. 
# You could spell that as: 

# conda install 'bar-lib>1.3.4,<
# 1.1' For this exercise, 
# install the latest compatible 
# version of attrs that is 
# later than version 16, but 
# earlier than version 17.3. 
# Which version gets installed?
# R/ ---->17.2.0



# ### --------------------------------------------------------
# # # ------>>>>Update a conda package
conda update pandas
y


# ### --------------------------------------------------------
# # # ------>>>> Remove a conda package
conda remove pandas
y


# ### --------------------------------------------------------
# # # ------>>>> Search for available package versions?
conda search attrs


# ### --------------------------------------------------------
# # # ------>>>>Find dependencies for a 
# package version? The conda 
# search package_name --info 
# command reports a variety of 
# details about a specific 
# package. The syntax for 
# specifying just one version 
# is a little bit complex, but 
# prefix notation is allowed 
# here (just as you would with 
# conda install). 

# For example, running conda 
# search cytoolz=0.8.2 --info 
# will report on all available 
# package versions. As this 
# package has been built for a 
# variety of Python versions, a 
# number of packages will be 
# reported on. You can narrow 
# your query further with, 
# e.g.: 

# $ conda search 
# cytoolz=0.8.2=py36_0 --info 

# cytoolz 0.8.2 py36_0 <hr 
# />----------------- file 
# name   : cytoolz-0.8.2-
# py36_0.tar.bz2 name        : 
# cytoolz version     : 0.8.2 
# build string: py36_0 build 
# number: 0 channel     : https:
# //repo.anaconda.com/pkgs/free/osx-
# 64 size        : 352 KB 
# arch        : x86_64 
# constrains  : () 
# date        : 2016-12-23 
# license     : BSD 
# md5         : 
# cd6068b2389b1596147cc7218f0438fd 
# platform    : darwin 
# subdir      : osx-64 
# url         : https:
# //repo.anaconda.com/pkgs/free/osx-
# 64/cytoolz-0.8.2-
# py36_0.tar.bz2 dependencies: 
# python 3.6* toolz >=0.8.0 You 
# may use the * wildcard within 
# the match pattern. This is 
# often useful to match 
# 'foo=1.2.3=py36*' because 
# recent builds have attached 
# the hash of the build at the 
# end of the Python version 
# string, making the exact 
# match unpredictable. 

# Determine the dependencies of 
# the package numpy 1.13.1 with 
# Python 3.6.0 on your current 
# platform.
conda search numpy=1.13.1=py36* --info
# R/libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=3.6,<3.7.0a0, mkl >=2018.0.0,<2019.0a0, and blas * mkl
