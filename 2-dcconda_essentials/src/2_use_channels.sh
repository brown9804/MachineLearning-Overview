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
