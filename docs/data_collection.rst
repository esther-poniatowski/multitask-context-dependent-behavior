Data Collection
===============

.. _data-collection:


Several servers are involved in the ETL 
All the raw data for the analysis has to be centralized in a secure server from the Laboratory of
Perceptual Systems (LSP) at Ecole Normale Supérieure (Paris).

The original data is collected from two sources:

- Data from Elguelda2019 is stored on collaborators' servers in the Maryland. 
- Data from Bagur2018 is already stored in another LSP server, referred to as the "BiggerGuy".

All the data should be gathered in a single location on the volume ``data4`` in the Network Attached
Storage (NAS) of the . This storage is a Network Share, i.e. a shared folder which allows multiple
users to access and manage files remotely.

Additionally, some ETL operations have to be performed from another LSP computer, referred to as "Mareesfreestyle".




Recovery from Maryland Servers
------------------------------

This process has to be performed from the LSP computer "Marrsfreestyle" (see :ref:`data-servers`).
TODO: Explain why. 

1. Set up the Virtual Private Network (VPN) with the Maryland server:

    1. Launch Global Protect (i.e. reinitialize it) TODO: What is global protect? What is the
       effect of "launching it"? What are the respective effects of accessing it from the webpage,
       and of running the command line?
       
       `Link text <https://terpware.umd.edu/Linux/Title/4010>`_ 
       Webpage of the Maryland VPN : `https://terpware.umd.edu/Linux/Title/4010`

       .. code-block:: sh

            globalprotect launch-ui

        Portal : `access.umd.edu` TODO: What is this?

    2. Enter the login and password of Shihab Shammah.

    3. Wait for the approval from Shihab Shammah, from a notification received on his phone.


2. Mount the source and destination servers:

    1. Mount the Maryland server in the directory ``/media/haka``:
    
       .. code-block:: sh 

            sudo mount -o username=yves //haka.isr.umd.edu/data /media/haka
    
       Password: abcd1234

    2. Mount the local volume ``data4`` into the directory ``/auto/``:

       .. code-block:: sh 
        
            sudo mount /auto/data4

3. Copy the data from the Maryland to the NAS server:

    1. Navigate to the directory containing the transfer codes:

    .. code-block:: sh 
    
        cd ~/Documents/esther-transfer-MDdata TODO: Correct this path.


    2. Launch a persistent session to ensure TODO: Why? 

    .. code-block:: sh
        
        byobu

    3. Run the transfer pipeline:
    
    .. code-block:: sh
        
        TODO: Which code?


    4. Press `F6` to detach the session, while the data is being copied.



DATA TRANSFER FROM BAGUR TO THE LOCAL NAS
-----------------------------------------

WARNING : Contrary to Elguelda, this procedure involves gathering data which is split in several locations (auto/data/ and data4/2018/Bagur/).

1) a) Log on LSP computer (see README > DISTANT WORK in the parent directory)
   b) Navigate to the directory containing the codes for transfer :

    `cd ~/Documents/esther-transfer-MDdata`

    NOTE : This step has to be performed from this computer because data1 is mounted in auto/.
    The path specified in UTILS_GLobalVariables.py are relative to this computer.

STEP 1 : Unpack data from 2018_Bagur into their respective directories (IDs of sites).
1) Set True to STEP 2 in TRANSFER_Bagur.py.
2) Run TRANSFER_Bagur.py
3) Note the names of the sessions raising an error message :
'ERROR : neither {spikefile} nor {spikefile_red} in {sub_dir_path}/Data_Raw/'
Report them in MISSING_SESSIONS in TRANSFER_Bagur.py.


STEP 2 : Complete the missing data (especially .m files) from the volume data in the NAS.
1) Mount the local volume data into the directory /auto/
sudo mount /auto/data
2) Mount the local volume data4 into the directory /auto/
sudo mount /auto/data4
3) Set True to STEP 3 in TRANSFER_Bagur.py.
4) Run TRANSFER_Bagur.py



