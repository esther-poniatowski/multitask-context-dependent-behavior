.. _install: 

Installation Guide
==================

This installation workflow creates:

- A *working directory* containing all the project files: documentation (HTML), packages, scripts, notebooks...
- A new *virtual environment* dedicated to the project, which automatically handles the installation of the Python interpreter and the dependencies. 

Guide to use this framework :ref:`usage`.


Requirements
------------

The installation process resorts to the following tools:

- `Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ 
- `Git <https://git-scm.com/downloads>`_ (optional)

Please ensure that they are correctly installed and configured on the system before proceeding with the installation steps.

.. warning::
   **Conda** is used as the primary package and environment management system (instead of other tools commonly used in Python environments such as ``pip``, ``setuptools``, ``venv``).


Step 1 - Create the Project's Working Directory
-----------------------------------------------

**Option 1: From a command line** [requires ``git``]

1. Navigate to the system location where the project working directory should reside.
2. Run the following git command:

.. code-block:: bash

    git clone https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git

**Option 2: From GitHub's interface** [does not require ``git``]

1. Open a web browser and navigate to the GitHub page of the repository: https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
2. Click on the green **Code** button, Choose **Download ZIP** from the dropdown menu.
3. Extract the ZIP file to the system location where the project working directory should reside.

In both cases, the entire project directory named ``multitask-context-dependent-behavior`` is created into the desired location. 


Step 2 - Create a Virtual Environment
-------------------------------------

.. code-block:: bash

    conda env create -f environment.yml

A new conda environment is created under the name ``mtcdb``.

.. note:: 
   Dependencies are stored separately from the project directory, in the Conda directory designated for the system.


Step 3 - Verify Installation
----------------------------

1. Activate the environment

.. code-block:: bash

    conda activate mtcdb

2. Check the version number

.. code-block:: bash

    python -c "import mtcdb; print(mtcdb.__version__)"

Expected outcome in the terminal : Number of the downloaded version of the project.

3. Deactivate the environment to return to the base environment

.. code-block:: bash

    conda deactivate
