Installation
============

**For Linux**
-------------

Environment set up
~~~~~~~~~~~~~~~~~~

To setup an environment to run ChemicalDice you can install miniconda
using command.

.. code:: bash

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

Follow the prompts after above commands to install conda. Make a
separate environment named chemdice using the conda create

.. code:: bash

   conda create -n chemicaldice python=3.9
   conda activate chemicaldice

Install packages
~~~~~~~~~~~~~~~~

To use the **ChemicalDice** package, you need to install it along with
its dependencies. You can install ChemicalDice and its dependencies
using the following commands:

.. code:: bash

   pip install -i https://test.pypi.org/simple/ ChemicalDice==0.7.2
   pip install multitasking==0.0.11 pandas==2.0.3 scikit-learn==1.2.2 seaborn==0.13.1 tqdm==4.66.4 xgboost==2.0.3 psutil==6.0.0
   pip install rdkit==2023.9.6 signaturizer==1.1.14 descriptastorus==2.6.1 mordred==1.2.0 tensorly==0.8.1 transformers==4.40.1
   pip install --upgrade tensorflow==2.15
   conda install conda-forge::openbabel
   conda install conda-forge::cpulimit

The pytorch package need to installed according to your versions of cuda
( computers with GPU ) and for computers with CPU only use the last
command.

.. code:: bash

   # ROCM 5.7
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
   # CUDA 11.8
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
   # CPU only
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

**For windows**
---------------

.. _environment-set-up-1:

Environment set up
~~~~~~~~~~~~~~~~~~

To setup an environment to run ChemicalDice you have to get three
things.

1. You can install miniconda using the installer given in the following
   web page https://docs.anaconda.com/miniconda/miniconda-install/
2. You. also need to MOPAC software which you can download and install
   using latest installer. https://github.com/openmopac/MOPAC/releases.
3. For windows system you also need to keep 3dmorse.exe for quantum
   descriptor in your current directory. You can download 3dmorse.exe
   from this GitHub repository
   https://github.com/devinyak/3dmorse/blob/master/3dmorse.exe.

.. _install-packages-1:

Install packages
~~~~~~~~~~~~~~~~

.. code:: bash

   pip install -i https://test.pypi.org/simple/ ChemicalDice==0.7.1
   pip install multitasking==0.0.11 pandas==2.0.3 scikit-learn==1.2.2 seaborn==0.13.1 tqdm==4.66.4 xgboost==2.0.3 psutil==6.0.0
   pip install rdkit==2023.9.6 signaturizer==1.1.14 descriptastorus==2.6.1 mordred==1.2.0 tensorly==0.8.1 transformers==4.40.1
   pip install --upgrade tensorflow==2.15
   conda install conda-forge::openbabel

The pytorch package need to installed according to your versions of cuda
( computers with GPU ) and for computers with CPU only use the last
command.

.. code:: bash

   # CUDA 11.8
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
   # CPU only
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
