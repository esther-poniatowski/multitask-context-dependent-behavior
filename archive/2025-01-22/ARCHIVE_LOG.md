# Archive Log

## Archive Details

- Date of archival: 2025-01-22
- Last commit hash on `main`: 978a68c24ba56c448fe3f373de0b8fff2b32069b

## Reason for Archival

Extract legacy code to clear the main branch. 

The code in this directory will not be used anymore as such in the main branch. It is kept for
reference and potential reuse in future projects. 

## Contents

`tasks/environment/`
- Original Location: `src/tasks/environment/`
- Notes: Bash scripts for setting up the conda environment and define environment variables.
- Future use: Some parts of those scripts might serve as inspiration for implementing a simpler
  setup process in the main branch, or to create a dedicated package for general environment setup.

`tasks/network/`
- Original Location: `src/tasks/network/`
- Notes: Bash and Python scripts to interact with a remote server. 
    - `connect.sh`: Utilities and executable to establish an SSH connection to a server.
    - `mount_data.sh`: Script to mount a Mount CIFS on the remote server. 
    - `transfer.py`: Utilities and executable to transfer files between the servers in both
      directions, according to a configuration file ("sync-map").
    
`test_tasks/`
- Original Location: `tests/test_tasks/`
- Notes: Test scripts for the tasks module.
    
`ops/`
- Original Location: `./ops/`
- Notes: Bash scripts for running key operations of the project. The `ops/` directory used to be
  added to the `PATH` by a post-activation script in the conda environment. Scripts were named
  without extension to be executable from the command line.
  - `build_docs`: Script to build the documentation in HTML with sphinx.
  - `connect_hub`: Script to connect to the remote server. 
  - `deploy_to_hub`: Script to deploy the project to the remote server.
  - `retrieve_from_hub`: Script to run the tests.
  - `test`: Script to run the tests. Not useful anymore since it can be performed from the VS Code
    interface.

`config/`
- Original Location: `./config/`
- Notes: Configuration files for the deployment to the remote server.
  - `dir_deploy.yml`: Directory structure to establish on the remote server.
  - `map_deploy.yml`: Synchronization map used by the `transfer.py` script to transfer the scripts
    and configuration files to the remote server.
  - `map_retrieve.yml`: Synchronization map used by the `transfer.py` script to retrieve the data
    files from the remote server.


# Relevant Files

To transfer to the `data-etl` branch:
- `tasks/network/mount_data.sh`: Script to mount a Mount CIFS on the remote server (to be run on the
  server side).

To integrate or adapt in the main branch:
- To setup the conda environment: Take inspiration from the scripts in `tasks/environment/`.
- To automate the documentation build: Integrate a similar script to `ops/build_docs`.
- To automate the connection to the remote server: Integrate a similar script to `ops/connect_hub`
  and take inspiration from `tasks/network/connect.sh`.
- To recover data from the remote server: Take inspiration from `tasks/network/transfer.py`,
  `retrieve_from hub`.

