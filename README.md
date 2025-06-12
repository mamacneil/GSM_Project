# GSM_Project
Codebase for the Global Shark Meat project

## Setup

We provide two sets of instructions to install the required packages needed to run these notebooks. The first will rely on [Mamba](https://mamba.readthedocs.io/en/latest/index.html), and the second will only use the standard [pip python package](https://packaging.python.org/tutorials/installing-packages/).

### Instructions for Mamba

[Mamba](https://mamba.readthedocs.io/en/latest/index.html) is a fast and convenient package manager that can create isolated python environments, install python and non-python packages and libraries, and greatly simplifies the process of setting up reproducible working environments. To continue with these instructions, please **make sure to install [mamba-forge](https://github.com/conda-forge/miniforge#mambaforge) on your system (pay attention to the link for each platform)**.

The next step is to clone or download the materials from this repository on your computer. Run the git clone command:

```
git clone https://github.com/mamacneil/Global_Shark_Meat.git
```

The repository has a file called `environment.yml` (or `environment-MacM1.yml` if you're using an M1 Mac), that includes a list of all the packages you need to install. If you run:

```
mamba env create -f environment.yml
```

from the root directory, it will create the environment for you and install all of the packages listed. This environment can be enabled using:

```
conda activate gsm-project
```

(yes, this is `conda`, not `mamba`. It's the only time when you need to do that. All the other commands will start with `mamba`)

Then you need to register this environment to be usable in jupyter lab by running this:

```
python -m ipykernel install --user --name gsm-project
```

Then, you can start jupyter lab to access the materials:

```
jupyter lab
```

### Instructions for pip

Sometimes, using Mamba is not an option. Don't worry, to run these notebooks you won't need much. You will have to first [install python and pip](https://www.python.org/downloads/) on your system. Then, you will need to install [graphviz](https://www.graphviz.org) on your system.

You can work in your system level python, but we strongly recommend that you isolate your environment by [creating a new virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) using the python provided `venv` tool.

```bash
python -m venv /path/to/new/virtual/environment
```

You will then have to activate your environment to install contents into it by running

```bash
source <venv>/bin/activate
```

Refer to [this link for instructions for Windows users](https://docs.python.org/3/library/venv.html#creating-virtual-environments)

Then, to install the required packages you will have to run:

```bash
python -m pip install -r requirements.txt
```

Then you need to register this environment to be usable in jupyter lab by running this:

```
python -m ipykernel install --user --name gsm-project
```

You can then start jupyter lab to access the materials by executing

```bash
jupyter lab
```


