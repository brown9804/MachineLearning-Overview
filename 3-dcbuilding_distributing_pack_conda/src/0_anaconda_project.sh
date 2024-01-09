# In the following exercises 
# you'll learn how to create, 
# run, and share Anaconda 
# Projects. 

# Which of the following CANNOT 
# be achieved with an Anaconda 
# Project?
# # Develop a project on your workstation and deploy it to a production web server.
# Automatically install the correct version of each required package before running a command.
# Provide an on-line editing environment for multiple users to develop code.
# Define datasets to be downloaded before running commands.
# R/ Provide an on-line editing environment for multiple users to develop code.


# ### --------------------------------------------------------
# # # ------>>>> Install Anaconda-Project
conda install anaconda-project -y


# ### --------------------------------------------------------
# # # ------>>>>Prepare and run a project command
anaconda-project list-packages
anaconda-project prepare
anaconda-project list-commands
anaconda-project run search-names Belinda


# # ### --------------------------------------------------------
# # # # ------>>>>Anaconda Project specification file
# The core 
# of an Anaconda Project is a 
# YAML file containing a 
# specification of the conda 
# packages, commands, and 
# downloads that make up the 
# Project. 

# The YAML file is called 
# anaconda-project.yml and each 
# separate project you create 
# will be in it's own 
# subdirectory containing a 
# distinct anaconda-project.yml 
# file for that project. 

# In the terminal you will see 
# that we have navigated to the 
# babynames directory. Use ls 
# to inspect the contents of 
# this directory. Further, 
# inspect the anaconda-
# project.yml file. You should 
# see YAML tags for packages, 
# commands, and downloads. 

# Choose the correct command 
# below that was used when you 
# executed anaconda-project run 
# search-names <NAME> in the 
# previous exercise. You can 
# use tools like nano, vim, 
# emacs, cat, less, or more to 
# read the file
# R/--> python main.py



# # ### --------------------------------------------------------
# # # # ------>>>>Initialize a new project
mkdir mortgage_rates
cd mortgage_rates
anaconda-project init
nano anaconda-project.yml
# adding -> description: Forecast 30-year mortgage rates in the US


# # # ### --------------------------------------------------------
# # # # # ------>>>>Anaconda Project commands As 
# you have seen Anaconda 
# Projects provide reproducible 
# execution of data science 
# assets. The babynames project 
# defined a command-line-
# interface (CLI) command to 
# analyze yearly trends. 

# When defining a command line 
# tool in Anaconda Project the 
# type unix or windows should 
# be used. Typically both are 
# defined, where unix is the 
# full command as run in Bash 
# and windows is the full 
# command as run in the Windows 
# Shell. 

# Anaconda Projects can support 
# four types of commands: 

# Unix commands: shell-based 
# commands that run on Mac or 
# Unix systems Windows 
# commands: Windows shell 
# commands Bokeh App: Run bokeh-
# server with a given Python 
# script Jupyter Notebook: 
# Launch Jupyter Notebook with 
# the specified notebook file 
# For both Unix and Windows 
# commands any arbitrary 
# command can be run. These 
# could be OS-specific tools or 
# Python scripts provided with 
# the Project. The Conda 
# environment defined in 
# anaconda-project.yml is 
# created and activated 
# automatically when running a 
# command. 

# Commands are added to 
# projects using the anaconda-
# project add-command command. 
# Projects can have any number 
# of commands defined. 

# Which of the following tasks 
# are not supported by Anaconda 
# Project commands?
# Launch a rich web dashboard built with Bokeh
# Start a light-weight REST API built with Python
# Launch a graphical user interface (GUI) tool
# R/ None of the above


# # # ### --------------------------------------------------------
# # # # # ------>>>>Add packages and commands
anaconda-project add-packages pandas
anaconda-project add-download MORTGAGE_RATES https://goo.gl/jpbAsR
nano forescast.py #-> change MORTGAGE_RATES = os.environ["MORTGAGE_RATES"]
anaconda-project add-command --type unix forecast "python forecast.py"
anaconda-project run forecast


# # # ### --------------------------------------------------------
# # # # # ------>>>>Locking package versions
anaconda-project lock


# # # ### --------------------------------------------------------
# # # # # ------>>>>Sharing your project
anaconda-project archive ../mortgage_rates.zip
anaconda login --username datacamp-student --password datacamp1
anaconda upload ../mortgage_rates.zip -t
