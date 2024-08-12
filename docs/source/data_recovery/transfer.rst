Data Transfer Across Servers
============================

.. _data-transfer:

Servers
-------
The original data is stored in two locations:

- Data from Elguelda2019 : Maryland server
- Data from Bagur2018 : BiggerGuy

All the data should be gathered in a single location on the volume ``data4`` in the Network Attached
Storage (NAS) of the MARRSFREESTYLE. This storage is a Network Share, i.e. a shared folder which allows multiple
users to access and manage files remotely.

Some transfer operations are performed from another LSP computer.


Pre-Requisites
--------------

1. From the root of the project, navigate to the directory containing the codes for transfer :

   .. code-block:: sh

      cd server-transfer

2. Share the SSH keys to the remote servers (if not already done):

   .. code-block:: sh

        make send-keys


Connecting to the Servers
-------------------------

Connect to the BiggerGuy:

   .. code-block:: sh

      make connect_biggerguy


Mounting Data File Systems
--------------------------

To access the NAS data directly from the home directory of the user on the BiggerGuy, the volume
``data4`` from the NAS has to be mounted in the sub-directory of the same name in the home directory.
This operation has to be performed occasionally, especially after a reboot of the computer.

.. code-block:: sh

    make mount_data4


DIrectory Trees
---------------

The different servers involved in this step should contain the following directory trees :

.. code-block:: plaintext

    LSPAUD1
    ├── data4
    │   └── MDdata
    │       ├── 2019_Elguelda
    │       │   └── .mat  ── Names the units analyzed in Elguelda2019 (provided by Diego Elguelda by mail).
    │       └── 2018_Bagur  ── Data analyzed in Bagur2018 (provided by Sophie Bagur by mail).
    │           ├── ClickRateDiscrimination_Aversive_A1
    │           │   └── Data_PSTH  ── WARNING: Data_Raw not provided.
    │           ├── ClickRateDiscrimination_Aversive_PFC
    │           │   ├── Data_PSTH
    │           │   └── Data_Raw
    │           │       ├── spk.mat
    │           │       ├── .m
    │           │       └── ...
    │           ├── ClickRateDiscrimination_Naive_A1
    │           │   └── ... (idem)
    │           └── ToneDetect_Aversive_A1
    │               └── ... (idem)


NOTE : This is the content of the directory "DataTINS" provided by Sophie Bagur,
whose other paradigms are removed.
The directory "DataBagur" is not included, because it contains redundant data
(and only PSTH binned at 100 ms).
The full data is conserved on the local computer.



/!\ STEPS TO PERFORM PRIOR TO DATA TRANSFER
-------------------------------------------


1) Copy the codes from the local computer to LSP computer :
    `scp /home/esther/Documents/These/Codes/Codes_Transfer/{TRANSFER_Elguelda.py,TRANSFER_Bagur.py,TRANSFER_utils.py}  $(USER_LSP)@129.199.80.205:~/Documents/esther-transfer-MDdata && scp /home/esther/Documents/These/Codes/UTILS_GlobalVariables.py  $(USER_LSP)@129.199.80.205:~/Documents/esther-transfer-MDdata`

2) Copy the codes from the local computer to the BiggerGur :
    `scp /home/esther/Documents/These/Codes/{UTILS_GlobalVariables.py,UTILS_ToolFunctions.py,UTILS_SaveRecover.py} esther@129.199.80.162:/mnt/working2/esther && scp /home/esther/Documents/These/Codes/Codes_Transfer/{TRANSFER_List_Sites.py,TRANSFER_utils.py,UTILS_List_Sites.py} esther@129.199.80.162:/mnt/working2/esther/Codes_Transfer`

3) Generate the lists of sites from the BiggerGuy :
    a) Connect to the BiggerGuy
    b) Move to Codes_Transfer/
    c) Run TRANSFER_List_Sites.py

NOTE : This step has to be done from the BiggerGuy, because it requires modules that are not installed on LSP computer (especially scipy to open .mat files in Python). Those modules are available in the environment at /mnt/working2/esther/miniconda3.

1) Copy the lists of sites from the BiggerGuy to LSP computer :
    a) Connect to the BiggerGuy
    b) Copy the files

    `scp /mnt/working2/esther/Data_Meta/{SITES_Elguelda.csv,SITES_Bagur.csv}  $(USER_LSP)@129.199.80.205:~/Documents/esther-transfer-MDdata`
NOTE : Those files are not necessary on the local computer, because all the information about sites and sessions will be available in the dataframes UNITSinfo, SESSIONSinfo etc, built on the BiggerGuy.


DATA TRANSFER FROM MARYLAND SERVERS TO THE LOCAL NAS
----------------------------------------------------

1) Log on LSP computer

2) Set up the VPN with the Maryland.

    a) Launch Global Protect (i.e. reinitialize it)
    Webpage of the Maryland VPN : https://terpware.umd.edu/Linux/Title/4010
    `globalprotect launch-ui`
    Portal : access.umd.edu

    b) Login & Pass of Shihab

    c) Shihab receives a notification on his phone and authorizes the connection.
    => VPN open

3) Mount the servers (see CONFIGURATION OF MOUNT POINTS below if problems).

    a) Mount the Maryland server in the directory /media/haka
    `sudo mount -o username=yves //haka.isr.umd.edu/data /media/haka`
    Password: abcd1234

    b) Mount the local volume data4 into the directory /auto/
    sudo mount /auto/data4

4) Copy the data from the Maryland in the local server.

    a) Navigate to the directory containing the codes for transfer :
    `cd ~/Documents/esther-transfer-MDdata`

    b) In TRANSFER_Elguelda.py, set True and False to execute only the desired steps.

    c) Launch a byobu session and run the code.
    `byobu`
    `python TRANSFER_Elguelda.py`

    Press F6 to detach the session.
    => The data is being copied.



DATA TRANSFER FROM BAGUR TO THE LOCAL NAS
-----------------------------------------

WARNING : Contrary to Elguelda, this procedure involves gathering data which is split in several locations (auto/data/ and data4/2018/Bagur/).

0) a) Log on LSP computer (see README > DISTANT WORK in the parent directory)
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



CONFIGURATION OF MOUNT POINTS
-----------------------------
In the file /ect/fstab :

NOTE : The file fstab should be parametrized to that mounting is automatic (simplified) when running sudo mount ...
