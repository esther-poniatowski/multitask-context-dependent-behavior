# File Systems

## Mounting Data File Systems

To access the NAS data directly from the home directory on the BiggerGuy, mount the volume `data4`
in the corresponding sub-directory. This operation is required after system reboots.

```sh
make mount_data4
```

:::{note}
Configure `/etc/fstab` to enable automatic mounting with `sudo mount ...`  
:::

## Directory Structures

The servers involved in this step should be organized as follows:

### LSPAUD1

```tree
├── data4
│   └── MDdata
│       ├── 2019_Elguelda
│       │   └── .mat    # Units analyzed in Elguelda2019 (provided by Diego Elguelda)
│       └── 2018_Bagur  # Data from Bagur2018 (provided by Sophie Bagur)
│           ├── ClickRateDiscrimination_Aversive_A1
│           │   └── Data_PSTH  # WARNING: Data_Raw not provided
│           ├── ClickRateDiscrimination_Aversive_PFC
│           │   ├── Data_PSTH
│           │   └── Data_Raw
│           │       ├── spk.mat
│           │       └── .m
│           ├── ClickRateDiscrimination_Naive_A1  # Same structure
│           └── ToneDetect_Aversive_A1  # Same structure
```

:::{note}  
This reflects the "DataTINS" directory provided by Sophie Bagur (keeping only relevant paradigms). 
The directory "DataBagur" is not included, because it contains redundant data (and only PSTH binned at 100 ms).
Full data remains on local machines.  
:::

### Ready for Processing

Raw and processed files are stored consistently with the following conventions:

```tree
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
