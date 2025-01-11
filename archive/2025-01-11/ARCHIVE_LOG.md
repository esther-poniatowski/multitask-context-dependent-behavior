# Archive Log

## Archive Details

- Date of archival: 2025-01-11
- Last commit hash on `main`: 3b79343c119c7e44b8d6d8eb14585958164c7fbd

## Reason for Archival

Simplification of the setup process on the main branch: the new set up process only includes a few
steps that can be performed manually and are documented in the `README.md` file. 

The contents of the archive serve for reference if a similar implementation is needed for a more
complex setup process in the future.

## Contents

`setup/`
- Original Location: `./setup/`
- Notes: The `setup` directory contained scripts and configurations that were used to initialize the
  project workspace by running automated bash and conda commands. The `hub` subdirectory contained
  hub-specific configurations to adapt the set up process with similar commands on the data server.

