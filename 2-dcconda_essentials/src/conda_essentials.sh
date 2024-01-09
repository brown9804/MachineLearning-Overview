# ### --------------------------------------------------------
# # # ------>>>>Channels and why are they 
# needed? All Conda packages 
# we've seen so far were 
# published on the main or 
# default channel of Anaconda 
# Cloud. A Conda channel is an 
# identifier of a path (e.g., 
# as in a web address) from 
# which Conda packages can be 
# obtained. Using the public 
# cloud, installing without 
# specifying a channel points 
# to the main channel at https:
# //repo.anaconda.com/pkgs/main; 
# where hundreds of packages 
# are available. Although 
# covering a wide swath, the 
# main channel contains only 
# packages that are (
# moderately) curated by 
# Anaconda Inc. Given finite 
# resources and a particular 
# area focus, not all genuinely 
# worthwhile packages are 
# vetted by Anaconda Inc. 

# If you happen to be working 
# in a firewalled or airgapped 
# environment with a private 
# installation of Anaconda 
# Repository, your default 
# channel may point to a 
# different (internal) URL, but 
# the same concepts will apply. 

# Anyone may register for an 
# account with Anaconda Cloud, 
# thereby creating their own 
# personal Conda channel. This 
# is covered in the companion 
# course Conda for Building & 
# Distributing Packages (along 
# with creating and uploading 
# your own packages). For this 
# course, just understand that 
# many users have accounts and 
# corresponding channels. 

# Which description best 
# characterizes Conda channels?
# R/Channels are a means for a user to publish packages independently.



# ### --------------------------------------------------------
# # # ------>>>>Searching within channels If 
# a particular colleague or 
# other recognized user may 
# have published a package 
# useful to you, you can search 
# for it using the anaconda 
# search command. For example, 
# David Mertz, the principal 
# author of this course, has a 
# channel and Anaconda Cloud 
# account called davidmertz. 
# You can search his channel 
# using the command below; the 
# option --channel (or -c for 
# short) specifies the channel 
# to search. Particular users 
# may have published more niche 
# software you would like to 
# use; for example, colleagues 
# of yours may publish packages 
# of special use in your field 
# or topic area. 

# $ conda search --channel 
# davidmertz --override-
# channels --platform linux-64 
# Loading channels: done 
# Name                       
# Version                   
# Build  Channel 
# accelerate                 
# 2.2.0               
# np110py27_2  davidmertz 
# accelerate                 
# 2.2.0               
# np110py35_2  davidmertz 
# accelerate-dldist          
# 0.1                 
# np110py27_1  davidmertz 
# accelerate-dldist          
# 0.1                 
# np110py35_1  davidmertz 
# accelerate-gensim          
# 0.12.3             
# np110py27_96  davidmertz 
# accelerate-gensim          
# 0.12.3             
# np110py35_96  davidmertz 
# accelerate-skimage         
# 0.1.0                    
# py27_1  davidmertz accelerate-
# skimage         
# 0.1.0                    
# py35_1  davidmertz 
# constants                  
# 0.0.2                    
# py35_0  davidmertz 
# humidity                   
# 0.1              
# py36ha038022_0  davidmertz 
# textadapter                
# 2.0.0                    
# py27_0  davidmertz 
# textadapter                
# 2.0.0                    
# py35_0  davidmertz 
# textadapter                
# 2.0.0                    
# py36_0  davidmertz In this 
# case, the switch --override-
# channels is used to prevent 
# searching on default 
# channels. The switch --
# platform is used to select a 
# platform that may differ from 
# the one on which the search 
# is run (absent the switch, 
# the current computer's 
# platform is used). 

# The first search is unusual 
# in that it does not specify a 
# package name, which is more 
# typical actual use. For 
# example, you might want to 
# know which versions of the 
# package of textadapter for 
# the win-64 platform are 
# available for any version of 
# Python (assuming you know in 
# which channels to look): 

# $ conda search -c conda-
# forge -c sseefeld -c 
# gbrener --platform win-64 
# textadapter Loading channels: 
# done 
# Name                       
# Version                   
# Build  Channel 
# textadapter                
# 2.0.0                    
# py27_0  conda-forge 
# textadapter                
# 2.0.0                    
# py27_0  sseefeld 
# textadapter                
# 2.0.0                    
# py34_0  sseefeld 
# textadapter                
# 2.0.0                    
# py35_0  conda-forge 
# textadapter                
# 2.0.0                    
# py35_0  sseefeld 
# textadapter                
# 2.0.0                    
# py36_0  sseefeld 



# Based on the examples shown, 
# in which of the channels used 
# in the examples above could 
# you find an osx-64 version of 
# textadapter for Python 3.6?
conda search -c conda-forge -c sseefeld -c gbrener -c davidmertz --platform osx-64 textadapter
# R/davidmertz


# ### --------------------------------------------------------
# # # ------>>>>Searching across channels
anaconda search boltons
# 17.1.0 for me it's the right one 
# but answer is 20.0.0

# ### --------------------------------------------------------
# # # ------>>>> Default, non-default, and special channels
conda search -c conda-forge | grep conda-forge | wc -l
211557
# R/ About 100 thousand



# ### --------------------------------------------------------
# # # ------>>>> Installing from a channel
conda install --channel conda-forge youtube-dl
y
conda list


# ### --------------------------------------------------------
# # # ------>>>>Environments and why are they 
# needed? Conda environments 
# allow multiple incompatible 
# versions of the same (
# software) package to coexist 
# on your system. An 
# environment is simply a file 
# path containing a collection 
# of mutually compatible 
# packages. By isolating 
# distinct versions of a given 
# package (and their 
# dependencies) in distinct 
# environments, those versions 
# are all available to work on 
# particular projects or tasks. 

# There are a large number of 
# reasons why it is best 
# practice to use environments, 
# whether as a data scientist, 
# software developer, or domain 
# specialist. Without the 
# concept of environments, 
# users essentially rely on and 
# are restricted to whichever 
# particular package versions 
# are installed globally (or in 
# their own user accounts) on a 
# particular machine. Even when 
# one user moves scripts 
# between machines (or shares 
# them with a colleague), the 
# configuration is often 
# inconsistent in ways that 
# interfere with seamless 
# functionality. Conda 
# environments solve both these 
# problems. You can easily 
# maintain and switch between 
# as many environments as you 
# like, and each one has 
# exactly the collection of 
# packages that you want. 

# For example, you may develop 
# a project comprising scripts, 
# notebooks, libraries, or 
# other resources that depend 
# on a particular collection of 
# package versions. You later 
# want to be able to switch 
# flexibly to newer versions of 
# those packages and to ensure 
# the project continues to 
# function properly before 
# switching wholly. Or 
# likewise, you may want to 
# share code with colleagues 
# who are required to use 
# certain package versions. In 
# this context, an environment 
# is a way of documenting a 
# known set of packages that 
# correctly support your 
# project. 

# Which statement is true of 
# Conda environments?
# R/Conda environments allow for flexible version management of packages.




# ### --------------------------------------------------------
# # # ------>>>>Which environment am I using?
# When using conda, you are 
# always in some environment, 
# but it may be the default (
# called the base or root 
# environment). Your current 
# environment has a name and 
# contains a collection of 
# packages currently associated 
# with that environment. There 
# are a few ways to determine 
# the current environment. 

# Most obviously, at a terminal 
# prompt, the name of the 
# current environment is 
# usually prepended to the rest 
# of your prompt in 
# parentheses. Alternatively, 
# the subcommand conda env list 
# displays a list of all 
# environments on your current 
# system; the currently 
# activated one is marked with 
# an asterisk in the middle 
# column. The subcommands of 
# conda env (sometimes with 
# suitable switches) encompass 
# most of your needs for 
# working with environments. 

# The output of conda env list 
# shows that each environment 
# is associated with a 
# particular directory. This is 
# not the same as your current 
# working directory for a given 
# project; being "in" an 
# environment is completely 
# independent of the directory 
# you are working in. Indeed, 
# you often wish to preserve a 
# certain Conda environment and 
# edit resources across 
# multiple project directories (
# all of which rely on the same 
# environment). The environment 
# directory displayed by conda 
# env list is simply the top-
# level file path in which all 
# resources associated with 
# that environment are stored; 
# you need never manipulate 
# those environment directories 
# directly (other than via the 
# conda command); indeed, it is 
# much safer to leave those 
# directories alone! 

# For example, here is output 
# you might see in a particular 
# terminal: 

# (test-project) $ conda env 
# list # conda environments: # 
# base                     
# /home/repl/miniconda 
# py1.0                    
# /home/repl/miniconda/envs/py1.0 
# stats-research           
# /home/repl/miniconda/envs/stats-
# research test-
# project          *  
# /home/repl/miniconda/envs/test-
# project Following the example 
# above, what is the name of 
# the environment you are using 
# in the current session? Even 
# if you determine the answer 
# without running a command, 
# run conda env list to get a 
# feel of using that subcommand.
conda env list
#R/ course-project



# ### --------------------------------------------------------
# # # # ------>>>>What packages are installed 
# in an environment? (I) The 
# command conda list seen 
# previously displays all 
# packages installed in the 
# current environment. You can 
# reduce this list by appending 
# the particular package you 
# want as an option. The 
# package can be specified 
# either as a simple name, or 
# as a regular expression 
# pattern. This still displays 
# the version (and channel) 
# associated with the installed 
# package(s). For example: 

# (test-env) $ conda list 
# 'numpy|pandas' # packages in 
# environment at 
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
# specifying 'numpy|pandas', 
# these same two lines would be 
# printed, simply interspersed 
# with many others for the 
# various other installed 
# packages. Notice that the 
# output displays the filepath 
# associated with the current 
# environment first: in this 
# case, 
# /home/repl/miniconda/envs/test-
# env as test-env is the active 
# environment (as also shown at 
# the prompt). 

# Following this example, what 
# versions of numpy and pandas 
# are installed in the current (
# base/root) environment?
conda list 'numpy|pandas'
# R/ numpy=1.16.0; pandas=0.22.0


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
