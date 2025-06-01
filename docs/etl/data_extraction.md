# Data Extraction

*Performed from the Bigger Guy (see below).*

## Directory Structure on the Bigger Guy

```tree
    ├── /esther/working2/
    │   ├── UTILS_GlobalVariables.py                # General modules.
    │   ├── UTILS_ToolFunctions.py
    │   ├── UTILS_SaveRecover.py
    │   ├── Codes_Extract/                          # Specific modules.
    │   │   ├── EXTRACT_main.py
    │   │   ├── EXTRACT_utils.py
    │   │   ├── EXTRACT_spikes.m
    │   │   └── EXTRACT_exptevents.m
    │   ├── Codes_Transfer/                         # Other tool modules.
    │   │   └── UTILS_List_Sites.py
    │   ├── Data_Meta/                              # Location of the future CELLS.pkl, SESSIONS.pkl, PATHS.pkl, NOT_FOUND.pkl
    │   ├── Data_PSTH/                              # Location of the future SPIKES.pkl
    │   └── data4/MDdata/
    │       ├── Sites/                              # Example: avo052a/
    │       │   ├── MFiles (sessions)               # Example: avo052a_p_PTD.m
    │       │   └── SpikeFiles (sessions)           # Example: avo052a_p_PTD.spk.mat
    │       ├── 2019_Elguelda/
    │       │   ├── .mat files                      # Names the the units analyzed in Elguelda2019 (provided by Diego Elguelda by mail).
    │       └── 2018_Bagur/                         # Data analyzed in Bagur2018 (provided by Sophie Bagur by mail).
    │           ├── ClickRateDiscrimination_Aversive_A1/
    │           │   └── Data_PSTH/
    │           │   WARNING : Data_Raw not provided.
    │           ├── ClickRateDiscrimination_Aversive_PFC/
    │           │   ├── Data_PSTH/
    │           │   └── Data_Raw/
    │           │       ├── spk.mat files
    │           │       └── .m files
    │           ├── ClickRateDiscrimination_Naive_A1/
    │           │   IDEM
    │           └── ToneDetect_Aversive_A1/
    │               IDEM
    └── ...
```

## Procedure

1. **Copy the codes from the local computer to the Bigger Guy:**

    ```sh
    scp /home/esther/Documents/These/Codes/{UTILS_GlobalVariables.py,UTILS_ToolFunctions.py,UTILS_SaveRecover.py}
    esther@129.199.80.162:/mnt/working2/esther 
    scp /home/esther/Documents/These/Codes/Codes_Extract/{EXTRACT_main.py,EXTRACT_utils.py,EXTRACT_spikes.m,EXTRACT_exptevents.m}
    esther@129.199.80.162:/mnt/working2/esther/Codes_Extract 
    scp /home/esther/Documents/These/Codes/Codes_Transfer/UTILS_List_Sites.py
    esther@129.199.80.162:/mnt/working2/esther/Codes_Transfer
    ```

2. **Connect to the Bigger Guy:**

    ```xh
    ssh -X esther@129.199.80.162 cd Codes_Extract
    ```

3. **Generate the required information for the MATLAB scripts:**

    - In `EXTRACT_main.py`, set steps 1 and 2 to `True`.
    - Run:

        ```sh
        python EXTRACT_main.py
        ```

4. **Extract the spikes by running the MATLAB script `EXTRACT_spikes.m`:**

    - Ensure the variable `PATH_BAGUR` is set correctly to `2018_Bagur/`.

    **Method 1: From the graphical interface**

    ```sh
    matlab >> run('Codes_Extract/EXTRACT_spikes.m')
    ```

    **Method 2: From the command line**

    ```sh
    byobu matlab -batch "EXTRACT_spikes"
    ```

5. **Collect the spikes:**

    - In `EXTRACT_main.py`, set steps 3 and 4 to `True`.
    - Run:

        ```sh
        python EXTRACT_main.py
        ```

6. **Copy the generated data (sessions, spikes, not found) from the Bigger Guy to the local
   computer:**

    ```sh
    scp esther@129.199.80.162:/mnt/working2/esther/Data_PSTH/SPIKES.pkl /home/esther/Documents/These/Codes/Data_PSTH

    scp
    esther@129.199.80.162:/mnt/working2/esther/Data_Meta/{CELLS.pkl,CELLS_Elguelda.pkl,CELLS_Bagur.pkl,SESSIONS.pkl,SITES_Bagur.pkl,SITES_Elguelda.pkl,NOT_FOUND.pkl}
    /home/esther/Documents/These/Codes/Data_Meta
    ```

7. **Extract the experimental events by running the MATLAB script `EXTRACT_exptevents.m`:**

    **Method 1: From the graphical interface**

    ```sh
    matlab >> run('Codes_Extract/EXTRACT_exptevents.m')
    ```

    **Method 2: From the command line**

    ```sh
    byobu matlab -batch "EXTRACT_exptevents"
    ```

---

*Note: No need to transfer anything to the local computer, because `UNITSinfo`, `SESSIONSinfo`,
`TRIALSinfo`, `EVENTSinfo` will be generated from the Bigger Guy in the next step (see
`README_metadata.txt` in `Codes_Metadata/`).*
