
Data Servers
============

.. _data-servers:

This guide:

- Documents the distinct servers are involved in the ETL process. 
- Indicates how to connect to the central server to orchestrate ETL steps.

Servers Names and Roles
-----------------------

Each server is referenced under a specific name in this project.

- **BiggerGuy**: Central server of the Laboratory of Perceptual Systems (LSP) at Ecole Normale
  Supérieure (Paris). This server supports a Network Attached Storage (NAS), i.e. a shared folder
  which allows multiple users to access and manage files remotely. 
  
  Two volumes are relevant for the project:
  
  - `data4`: Destination location where all the raw data will be gathered.
  
  - `data1`: Source location which stores data from Bagur2018.

- **Maryland**: Collaborators' server in the Maryland. Source location which stores data from
  Elguelda2019.

- **Marresfreestyle**: LSP computer where some ETL operations have to be performed. TODO: What
  operations and why?

Credentials
-----------

The credentials to use to connect to each server have to be specified in `.env` files, named with
the server nickname. The required format is indicated in the template file located at [`config/template.env`](config/template.env). 

Those actual credentials are encrypted in an archive located at
[`config/credentials.tar`](config/credentials.tar). TODO: Create the archive.

.. _mount-points:  

Mount Points 
------------


Connecting to Servers
---------------------

The ETL steps have to be run from either 
1. Share the SSH keys to the remote servers (if not already done):

   .. code-block:: sh

        make send-keys


Connecting to the Servers
-------------------------

Connect to the BiggerGuy:

   .. code-block:: sh

      make connect
