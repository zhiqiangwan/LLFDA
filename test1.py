import os
import glob

processed_dataset_path = './processed_dataset_h5/SIM10K'
path = os.path.join(processed_dataset_path, '*.h5')
a = glob.glob(os.path.join(processed_dataset_path, '*.h5'))
print(a)