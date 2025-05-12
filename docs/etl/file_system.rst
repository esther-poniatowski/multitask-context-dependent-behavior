Mounting Data File Systems
--------------------------

To access the NAS data directly from the home directory of the user on the BiggerGuy, the volume
``data4`` from the NAS has to be mounted in the sub-directory of the same name in the home directory.
This operation has to be performed occasionally, especially after a reboot of the computer.

.. code-block:: sh

    make mount_data4

CONFIGURATION OF MOUNT POINTS
-----------------------------
In the file /ect/fstab :

NOTE : The file fstab should be parametrized to that mounting is automatic (simplified) when running sudo mount ...



Directory Structure
-------------------

The different servers involved in this step should contain the following directory trees :

.. code-block:: text

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



Raw and processed files are stored in the following directory structure:
```
/processed_data/
├── site1/
│   ├── session1.csv
│   ├── session2.csv
│   └── ...
├── site2/
│   ├── session1.csv
│   ├── session2.csv
│   └── ...
└── ...
```