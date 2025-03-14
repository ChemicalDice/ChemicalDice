Chemical Dice
=============

ChemicalDice presents an innovative paradigm shift in cheminformatics
and bioinformatics, leveraging advanced feature fusion methodologies to
transcend traditional data silos. By ingeniously amalgamating disparate
data modalities from chemical descriptors. Through a rich arsenal of
techniques including  Chemical Dice Integrator, PCA, ICA, IPCA, CCA,
t-SNE, KPCA, RKS, SEM, Tensor Decomposition, and PLSDA.
ChemicalDice unlocks novel insights by unraveling the intricate
interplay between chemical and biological entities within complex
datasets. Its intuitive interface and comprehensive toolkit empower
researchers to seamlessly navigate through the complexities of
integrative analyses.

Environment set up
==================

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
   pip install multitasking==0.0.11 pandas==2.0.3 scikit-learn==1.2.2 seaborn==0.13.1 tqdm==4.66.4 xgboost==2.0.3
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

   pip install -i https://test.pypi.org/simple/ ChemicalDice==0.6.9
   pip install multitasking==0.0.11 pandas==2.0.3 scikit-learn==1.2.2 seaborn==0.13.1 tqdm==4.66.4 xgboost==2.0.3
   pip install rdkit==2023.9.6 signaturizer==1.1.14 descriptastorus==2.6.1 mordred==1.2.0 tensorly==0.8.1 transformers==4.40.1
   pip install --upgrade tensorflow==2.15
   conda install conda-forge::openbabel
   pip install psutil

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


Calculation of descriptors
--------------------------

.. code:: python

   # create a directory for storing descriptors filefrom ChemicalDice 
   from ChemicalDice import smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
   import os
   os.mkdir("Chemicaldice_data")
   # download prerequisites for quantum, grover and ImageMol
   quantum.get_mopac_prerequisites()
   # input file containing SMILES and labels
   input_file = "your_file_name.csv"
   # preprocessing of smiles to different formats
   smiles_preprocess.add_canonical_smiles(input_file)
   smiles_preprocess.create_mol2_files(input_file)
   smiles_preprocess.create_sdf_files(input_file)
   # calculation of all descriptors
   quantum.descriptor_calculator(input_file, output_file="Chemicaldice_data/mopac.csv")
   Grover.get_embeddings(input_file,  output_file_name="Chemicaldice_data/Grover.csv")
   ImageMol.image_to_embeddings(input_file, output_file_name="Chemicaldice_data/ImageMol.csv")
   chemberta.smiles_to_embeddings(input_file, output_file = "Chemicaldice_data/Chemberta.csv")
   bioactivity.calculate_descriptors(input_file, output_file = "Chemicaldice_data/Signaturizer.csv")
   chemical.descriptor_calculator(input_file, output_file="Chemicaldice_data/mordred.csv")

Reading Data
------------

Define data path dictionary with name of dataset and csv file path. The
csv file should contain ID column along with features columns. Label
file should contain id and labels. If these columns not named id and
labels you can provide\ ``id_column`` and ``label_column`` argument
during initialization of ``fusionData``.

.. code:: python

   from ChemicalDice.fusionData import fusionData
   data_paths = {
      "Chemberta":"Chemicaldice_data/Chemberta.csv",
      "Grover":"Chemicaldice_data/Grover.csv",
      "mopac":"Chemicaldice_data/mopac.csv",
      "mordred":"Chemicaldice_data/mordred.csv",
      "Signaturizer":"Chemicaldice_data/Signaturizer.csv",
      "ImageMol": "Chemicaldice_data/ImageMol.csv"
   }

loading data from csv files and creating ``fusionData`` object.

.. code:: python

   fusiondata = fusionData(data_paths = data_paths, label_file_path="freesolv.csv", label_column="labels", id_column="id")

After loading data, you can use ``fusionData`` object to access your
data by ``dataframes`` dictionary in fusion data object. This is
important to look at the datasets before doing any analysis. For example
to get all dataframes use the following code.

.. code:: python

   fusiondata.dataframes

Data Cleaning
-------------

Common samples
~~~~~~~~~~~~~~

Keep only samples (rows) that are common across dataset. This is
important if there is difference in set of samples across datasets.

.. code:: python

   fusiondata.keep_common_samples()

Empty Features removal
~~~~~~~~~~~~~~~~~~~~~~

Features in data should be removed if there is higher percentage of
missing values. Remove columns with more than a certain percentage of
missing values from dataframes can solve this. The percentage threshold
of missing values to drop a column. ``threshold`` should be between 0
and 100. ``ShowMissingValues`` is function which prints the count of
missing values in each dataset.

.. code:: python

   fusiondata.ShowMissingValues()
   fusiondata.remove_empty_features(threshold=20)
   fusiondata.ShowMissingValues()

Imputation/Remove features
~~~~~~~~~~~~~~~~~~~~~~~~~~

Imputation of data if the data have low percentage of missing values.
``ImputeData`` is a function which takes a single argument which is
method to be used for imputation. The ``method`` can be “knn”, “mean”,
“mode”, “median”, and “interpolate”.

.. code:: python

   # Imputing values with missing valuesfusiondata.ShowMissingValues()
   fusiondata.ImputeData(method="knn")
   fusiondata.ShowMissingValues()

Data Normalization
------------------

Normalization/Standardization/Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data should be normalized before we proceed to fusion. There are three
functions which can be used for data normalization ``scale_data``,
``normalize_data`` and ``transform_data``. These functions takes single
argument that is type of scaling/normalization/transformation.

.. code:: python

   # Standardize data
   fusiondata.scale_data(scaling_type = 'standardize')

``scaling_type`` can be one of these ‘minmax’ , ‘minmax’ ‘robust’ or
‘pareto’

.. code:: python

   # Normalize data
   fusiondata.normalize_data(normalization_type ='constant_sum')

``normalization_types`` can be one of these ‘constant_sum’, ‘L1’ ,‘L2’
or ‘max’

.. code:: python

   # Transform data
   fusiondata.transform_data(transformation_type ='log')

``transformation_type`` can be one of these ‘cubicroot’, ‘log10’, ‘log’,
‘log2’, ‘sqrt’, ‘powertransformer’, or ‘quantiletransformer’.

**Data Fusion**
---------------

Data fusion will take all the data that is normalized in previous step
and make a single fused data. The ``fuseFeatures`` method can be used to
fuse the data and save it in a csv file. The fusion methods to use given
by methods argument. Methods available for fusing data are ‘CDI’, ‘pca’,
‘ica’, ‘ipca’, ‘cca’, ‘tsne’, ‘kpca’, ‘rks’, ‘SEM’ and ‘tensordecompose’.
The number of components to keep from different data in fusion can be
provided by ``n_components`` aggumrent. Reduced dimensions to use for
Chemical Dice Integrator can be provided by ``CDI_dim`` argument.
Argument ``save_dir`` can be used to specify directory for saving the
fused data. ``CDI_k`` a list representing the reduction in the number 
of nodes from the first layer to the second and so on for six feature 
vectors. ``CDI_epochs`` the number of epochs for training the autoencoder 
in the CDI method.


.. code:: python

   # fusing features in different data
   fusiondata.fuseFeatures(n_components=10,
                     methods= ['pca','tensordecompose','plsda','CDI'],
                     CDI_dim= [4096,8192],
                     CDI_k=[10, 7, 12, 5, 10, 6],
                     CDI_epochs=500,
                     save_dir = "ChemicalDice_fusedData")

**Evaluation of Fusion Methods**
--------------------------------

**Cross Validation**
~~~~~~~~~~~~~~~~~~~~

The method ``evaluate_fusion_model_nfold`` can perform n-fold cross
validation for the evaluation of fusion methods. It takes
the ``nfold`` argument for the number of folds to use for
cross-validation, the ``task_type`` argument for classification or
regression problems, and the ``fused_data_path`` directory that contains
the fused data as CSV files generated in the feature fusion step.

.. code:: python

   # Evaluate all models using 10-fold cross-validation for regression tasks
   fusiondata.evaluate_fusion_models_nfold(folds=10,
                                           task_type="regression",
                                           fused_data_path="ChemicalDice_fusedData")

Metrics for all the models can be accessed using
the ``get_accuracy_metrics`` method, which takes
the ``result_dir`` argument for the directory containing CSV files from
n-fold cross-validation. The outputs are
dataframes ``mean_accuracy_metrics`` and ``accuracy_metrics``, along
with boxplots for the top models for each fusion method saved
in ``result_dir``.

::

   ## Accuracy metrics for all models
   mean_accuracy_metrics, accuracy_metrics = fusiondata.get_accuracy_metrics(result_dir='10_fold_CV_results')

**Scaffold Splitting**
~~~~~~~~~~~~~~~~~~~~~~

The method ``evaluate_fusion_models_scaffold_split`` can perform
scaffold splitting for the evaluation of fusion methods. It takes the
arguments ``split_type`` (“random” for random scaffold splitting,
“balanced” for balanced scaffold splitting, and “simple” for just
scaffold splitting), ``task_type`` for “classification” or “regression”
problems, and the ``fused_data_path`` directory that contains the fused
data as CSV files generated in the feature fusion step.

.. code:: python

   # Evaluate all models using random scaffold splitting for regression tasks
   fusiondata.evaluate_fusion_models_scaffold_split(split_type="random",
                                                    task_type="regression",
                                                    fused_data_path="ChemicalDice_fusedData")

Metrics for all the models can be accessed using
the ``get_accuracy_metrics`` method, which takes
the ``result_dir`` argument for the directory containing CSV files from
scaffold splitting. The outputs are
dataframes ``test_metrics``, ``train_metrics``, and ``val_metrics``,
along with bar plots for the top models for each fusion method saved
in ``result_dir``.

.. code:: python

   ## Accuracy metrics for all models
   test_metrics, train_metrics, val_metrics = fusiondata.get_accuracy_metrics(result_dir='scaffold_split_results')
