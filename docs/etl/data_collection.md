# Data Collection

## Servers

Several servers are involved in the ETL process.

The raw data is collected from two sources:

- Data from Elguelda2019 is stored on collaborators' servers in Maryland.
- Data from Bagur2018 is already stored in another LSP server ("BiggerGuy").

All data is centralized in a secure server from the Laboratory of Perceptual Systems (LSP) at Ecole
Normale Supérieure (Paris). Is is stored on the volume ``data4`` in the Network Attached Storage
(NAS). This storage is a Network Share, i.e. a shared folder allowing remote file
access/management.

All data should be gathered in a single location on the volume `data4` in the Network Attached
Storage (NAS). This storage is a Network Share (shared folder allowing ).

Additionally, some transfer operations must be orchestrated from another LSP computer
("Mareesfreestyle").


## Recovery from Maryland Servers

This process must be performed from "Marrsfreestyle" (see {ref}`data-servers`).

<!-- TODO: Explain why -->

1. **Setting up a Virtual Private Network (VPN)**:

    1. Launch Global Protect (i.e. reinitialize it)

        Webpage of the Maryland VPN: <https://terpware.umd.edu/Linux/Title/4010>

            ```
            globalprotect launch-ui
            ```

        Portal : `access.umd.edu`

        <!-- TODO: Clarifications needed: What is global protect? What is the effect
       of "launching it"? What are the respective effects of accessing it from the webpage, and of
       running the command line? What is the portal? -->

    2. Enter Shihab Shammah's credentials.

    3. Wait for the approval from Shihab Shammah, from a notification received on his phone.

2. **Mounting the servers**:

    1. Mount the Maryland server in the directory `/media/haka`:

            ```sh
            sudo mount -o username=yves //haka.isr.umd.edu/data /media/haka
            ````

    2. Mount the local NAS `data4` into the directory `/auto/`:

            ```sh
            sudo mount /auto/data4
            ```

3. **Data transfer**:

    1. Navigate to the directory containing the transfer codes:

            ```sh
            cd ~/Documents/esther-transfer-MDdata
            ```

        <!-- TODO: Verify path -->

    2. Launch a persistent session: <!-- TODO: Specify why a persistent session is needed -->

            ```sh
            byobu
            ```

    3. Run the transfer pipeline: <!-- TODO: Specify code -->  

    4. Press `F6` to detach the session during copy.

## Data Transfer from Bagur to Local NAS

:::{warning} Data is split across two locations: `auto/data/` and `data4/2018/Bagur/` :::

1. **Preparation**:

    1. Log into LSP computer. <!-- TODO: Include link to instructions -->

    2. Navigate to transfer codes:

            ```sh
            cd ~/Documents/esther-transfer-MDdata 
            ```

        <!-- TODO: Verify path -->

        *Note: Requires mounted `data1` in `auto/`*

2. **Transfer**: <!-- TODO: Adapt to the new code -->
