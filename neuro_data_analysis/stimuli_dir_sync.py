#%%
# Alfa-13012021-003 this one is very sparse! maybe wrong unit id????
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

def copy_dir(stimuli_dir):
    target_dir = stimuli_dir.replace("N:", "S:")
    try:
        shutil.copytree(stimuli_dir, target_dir)
        print(f"Successfully copied from {stimuli_dir} to {target_dir}")
    except Exception as e:
        print(f"Error copying from {stimuli_dir} to {target_dir}: {e}")

ExpRecord_Hessian_All = pd.read_csv(r"ExpRecord_BigGAN_Hessian_tuning_ABCD_w_meta.csv")
ExpRecord_Evol_All = pd.read_csv(r"ExpRecord_BigGAN_Hessian_Evol_ABCD_w_meta.csv")
# for rowi, exprow in ExpRecord_Hessian_All.iterrows():
#     stimuli_dir = exprow.stimuli
#     # copy the stimuli dir to 
#     target_dir = stimuli_dir.replace("N:", "S:")
#     shutil.copytree(stimuli_dir, target_dir)
ExpRecord_All = pd.concat([ExpRecord_Hessian_All, ExpRecord_Evol_All])
# Assuming ExpRecord_Hessian_All is your pandas DataFrame
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all copy tasks to the executor
    futures = [
        executor.submit(copy_dir, row['stimuli'])
        for _, row in ExpRecord_All.iterrows()
    ]
    
    # Optionally, process results as they complete
    for future in as_completed(futures):
        try:
            future.result()  # This will re-raise any exceptions caught in copy_dir
        except Exception as e:
            print(f"Exception during copy: {e}")

print("All copy operations have been submitted.")


