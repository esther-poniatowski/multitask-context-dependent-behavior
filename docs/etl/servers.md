
# Data Servers

This guide:

- Documents the servers involved in the ETL process.
- Indicates how to connect to the central server for orchestrating ETL steps.

## Servers Aliases and Roles

Each server is referenced under a specific alias in this project.

- **BiggerGuy**: Central server in the Laboratory of Perceptual Systems (LSP) at École Normale
  Supérieure (Paris). This server supports a Network Attached Storage (NAS), i.e. a shared folder
  which allows multiple users to access and manage files remotely.

  Two volumes are relevant for the project:
  
  - `data4`: Final storage where all raw data will be gathered.
  - `data1`: Source location for Bagur2018 data.

- **Maryland**: Collaborators' server in the Maryland.
  Source location for Elguelda2019 data.

- **Marresfreestyle**: LSP computer for specific ETL operations.

## Credentials

Store credentials in server-specific in `.env` files, named by server. The required format is
indicated in the template file located at [`config/template.env`](config/template.env).

:::{warning}  
Actual credentials are encrypted in [`config/credentials.tar`](config/credentials.tar)  
:::

<!-- TODO: Create the archive -->

## Connecting to Servers

Codes implementing server connections are organized as follows:

```tree
project-root/
├── config/                    # Configuration files
│   ├── template.env           # Credential template
│   └── credentials.env        # Actual credentials (SSH_USER, SSH_HOST, SSH_PATH_KEY)
├── src/
│   └── remote_access/
│       ├── ssh_utils.sh       # SSH utilities (refactored)
│       └── ssh_connection.sh  # Connection logic (needs refactoring)
└── main.sh                    # Main entry point
```

1. Share the SSH keys to the remote servers (if not already done):

    ```sh
    send-keys <alias>
    ```

2. Connect to a remote server by its alias:

    ```sh
    connect <alias>
    ```

<!-- TODO: Implement those commands -->
