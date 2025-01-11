# Archive Log

## Archive Details

- Date of archival: 2025-01-04
- Last commit hash on `main`: 74927477635c8d623dd3d4958d8ee63bb15c5fbe

## Reason for Archival

Initialization of the archive branch for legacy files and directories.

At the indicated archive date, all the contents of the previous `archive` directory on the `main`
branch were moved to this new branch. 

> [!WARNING]
> The indicated archive date for the whole set of files does not correspond to the date when each
> individual file was archived itself. The date is used to group all the files that were archived
> before without archive log metadata.

> [!NOTE] It is not possible to retrieve the exact archive dates from the commit history, because
> the previous `archive` directory was not tracked before. It is not either possible to identify the
> date from file deletions, since archived files were usually *copied* from the main directory to
> the `archive` directory without deleting the original. 

## Contents

To be determined for each file in a global review of the archive.
Each file should be either:
- Kept in the archive for reference, with a documentation indicating its interest.
- Removed from the archive if it is no longer relevant.