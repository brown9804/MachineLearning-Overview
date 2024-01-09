# ### --------------------------------------------------------
# # # ------>>>>What packages are installed 
# in an environment? (II) It is 
# often useful to query a 
# different environment's 
# configuration (i.e., as 
# opposed to the currently 
# active environment). You 
# might do this simply to 
# verify the package versions 
# in that environment that you 
# need for a given project. Or 
# you may wish to find out what 
# versions you or a colleague 
# used in some prior project (
# developed in that other 
# environment). The switch --
# name or -n allows you to 
# query another environment. 
# For example, 

# (course-env) $ conda list --
# name test-env 'numpy|pandas' 
# # packages in environment at 
# /home/repl/miniconda/envs/test-
# env: # # 
# Name                    
# Version                   
# Build  Channel 
# numpy                     
# 1.11.3                   
# py35_0 
# pandas                    
# 0.18.1              
# np111py35_0 Without 
# specifying the --name 
# argument, the command conda 
# list would run in the current 
# environment. The output would 
# then be the versions of numpy 
# and pandas present in the 
# current environment. 

# Suppose you created an 
# environment called pd-2015 in 
# 2015 when you were working on 
# a project. Identify which 
# versions of numpy and pandas 
# were installed in the 
# environment pd-2015.
conda list -n pd-2015 'numpy|pandas'
# R/ numpy=1.16.4; pandas=0.22.0


# ### --------------------------------------------------------
# # # ------>>>> Switch between environments
conda activate course-env
conda activate pd-2015
conda deactivate


# ### --------------------------------------------------------
# # # ------>>>> Remove an environment
conda env remove -n deprecated

# ### --------------------------------------------------------
# # # ------>>>>Create a new environment
conda create -n conda-essentials attrs=19.1.0 cytoolz
y
conda activate conda-essentials
conda list


# ### --------------------------------------------------------
# # # ------>>>>Export an environment
conda env export -n course-env -f course-env.yml


# ### --------------------------------------------------------
# # # ------>>>>Create an environment from a shared specification
conda env create --file environment.yml
conda env create -f shared-config.yml


# ### --------------------------------------------------------
# # # ------>>>>Compatibility with different versions
cat weekly_humidity.py
python weekly_humidity.py
conda activate pd-2015
python weekly_humidity.py

# ### --------------------------------------------------------
# # # ------>>>>Updating a script
nano weekly_humidity.py
print(humidity.rolling(7).mean().tail(5))
# control + x
#  enter
python weekly_humidity.py
conda activate pd-2015
python weekly_humidity.py
