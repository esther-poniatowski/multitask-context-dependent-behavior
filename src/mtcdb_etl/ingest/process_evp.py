import os
import subprocess
import gzip
import shutil
import matlab.engine


# Step 2: Start MATLAB and set the path
eng = matlab.engine.start_matlab()
eng.cd(clone_dir, nargout=0)
eng.baphy_set_path(nargout=0)

# Step 3: Unzip the .evp.gz file
evp_gz_path = "/mnt/working2/esther/data4/MDdata/tan025b/tan025b01_p_FTC.evp.gz"
evp_path = evp_gz_path[:-3]

with gzip.open(evp_gz_path, 'rb') as f_in:
    with open(evp_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Step 4: Call MATLAB script to process the data
output_csv_path = evp_path.replace(".evp", ".csv")
eng.process_evp(evp_path, output_csv_path, nargout=0)

# Step 5: Call MATLAB script to process the data
output_csv_path = evp_path.replace(".evp", ".csv")
eng.process_evp(evp_path, output_csv_path, nargout=0)

# Step 6: Remove the locally unzipped file
print(f"Removing {evp_path}")
os.remove(evp_path)

print(f"CSV file saved to {output_csv_path}")
