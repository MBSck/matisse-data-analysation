# MATISSE-Data-Analysation

> The code in this repository has been made to streamline the data reduction with the MATISSE-pipeline

## Project Status
_Is currently being worked on_

## Table of Contents
* [Data-Reduction](#Data-Reduction)
* [Ressources](#Ressources)
* [Credit](#Credit)

# Data-Reduction
## Basic reduction
1. Edit `fn_call.sh` to have correct settings and directories
2. `bash fn_call.sh`

## Overnight reduction
1. Edit `fn_call.sh` to have correct settings and directories
2. `nohup bash fn_call.sh &`

# Ressources
## Jupyter notebook
1. ON astro-node `jupyter notebook --no-browser`
2. ON local computer `ssh -N -L localhost:8000:localhost:8888 astro-node11`
3. Then open browser to localhost:8000
4. (sometimes) copy/paste the token given in the astro-node terminal


## Credit
### Author
Jacob Isbell
### Contributors:
[Marten Scheuck](#https://github.com/MBSck/)
