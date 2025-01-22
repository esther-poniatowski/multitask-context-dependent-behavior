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
    - `mount_data.sh`: Script to mount a Mount CIFS on the remote server. TODO: Transfer this file
       in the `data-etl` branch, since it is aimed to be run on the server side.
    - `transfer.py`: Utilities and executable to transfer files between the servers in both
      directions, according to a configuration file ("sync-map"). TODO: Take inspiration from this
      file in the main branch to recover data from the remote server. 
    
`test_tasks/`
- Original Location: `tests/test_tasks/`
- Notes: Test scripts for the tasks module.
    

