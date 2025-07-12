# Conda Essentials

Costa Rica

[![GitHub](https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff)](https://github.com/)
[brown9804](https://github.com/brown9804)

March, 2021

----------

# What are packages and why are they needed? 
Conda packages are files containing a bundle of resources: usually libraries and executables, but not always. In principle, Conda packages can include data, images, notebooks, or other assets. The command-line tool conda is used to install, remove and examine packages; other tools such as the GUI Anaconda Navigator also expose the same capabilities. This course focuses on the conda tool itself (you'll see use cases other than package management in later chapters). 

Conda packages are most widely used with Python, but that's not all. Nothing about the Conda package format or the conda tool itself assumes any specific programming language. Conda packages can also be used for bundling libraries in other languages (like R, Scala, Julia, etc.) or simply for distributing pure binary executables generated from any programming language. 

One of the powerful aspects of conda—both the tool and the package format—is that dependencies are taken care of. That is, when you install any Conda package, any other packages needed get installed automatically. Tracking and determining software dependencies is a hard problem that package managers like Conda are designed to solve. 

A Conda package, then, is a file containing all files needed to make a given program execute correctly on a given system. Moreover, a Conda package can contain binary artifacts specific to a particular platform or operating system. Most packages (and their dependencies) are available for Windows (win-32 or win-64), for OSX (osx-64), and for Linux (linux-32 or linux-64). A small number of Conda packages are available for more specialized platforms (e.g., Raspberry Pi 2 or POWER8 LE). As a user, you do not need to specify the platform since Conda will simply choose the Conda package appropriate for the platform you are using. 

## Courses:

- [Installing Packages](./2-dcconda_essentials/src/1_install_pckgs.sh)
- [Utilizing Channels](./2-dcconda_essentials/src/2_use_channels.sh)
- [Working with Environments](./2-dcconda_essentials/src/3_work_envs.sh)
- [Case Study on Using Environments](./2-dcconda_essentials/src/4_use_envs.sh)

<!-- START BADGE -->
<div align="center">
  <img src="https://img.shields.io/badge/Total%20views-1022-limegreen" alt="Total views">
  <p>Refresh Date: 2025-07-11</p>
</div>
<!-- END BADGE -->
