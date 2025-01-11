# Archive Branch for Multi-Task Context-Dependent Decision Making 

This branch contains legacy files and directories that have been removed from the main development
branch. The purpose of this archive is to preserve past implementations for reference, rollback, or
future reuse without cluttering the active codebase.

> [!WARNING]
> Once moved to this branch, files become read-only.  
> This branch is not intended to be merged to the main branch.

## Archiving Criteria

Files and directories are moved to this branch if:
- They are no longer actively developed or used in the project.
- They have been superseded by newer versions.
- They were exploratory or experimental but were not pursued.
- They implement a relevant programming design that may serve as a model for future work.

## Metadata

Each archive item includes its own `ARCHIVE_LOG.md` file specifying:
- Date of archival
- Reason for archival
- Last active commit hash (optional)

## Structure

The archive is organized as follows:
- Top-level directories named by date: `YYYY-MM-DD/`.
- Subdirectories for specific features, modules, experiments, with a descriptive name.
- Each subdirectory preserves the original structure of the archived part of the project. Files
  retain their original paths inside the archived directories.
- Each subdirectory contains an `ARCHIVE_LOG.md` file.

```plaintext
archive-legacy/
├── README.md
├── 2023-01-01/
│   ├── ARCHIVE_LOG.md
│   ├── feature_alpha/
│   │   └── ...
│   └── module_beta/
│       └── ...
├── 2024-02-03/
│   ├── ARCHIVE_LOG.md
│   ├── experiment_x/
│   │   └── ...
│   └── library_y/
│       └── ...
└── ...
```
