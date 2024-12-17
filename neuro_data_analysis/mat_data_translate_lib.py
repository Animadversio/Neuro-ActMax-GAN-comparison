import numpy as np
from easydict import EasyDict as edict
import h5py

def hdf5_dataset_to_string(dset):
    # Read all data into a NumPy array
    arr = dset[()]  # This should be a 2D array of shape (N, 1)
    # Flatten to a 1D array of 16-bit code points
    code_points = arr.flatten()
    # Convert code points directly to characters and join
    # If these are standard Unicode code points, this works directly.
    # If the data is UTF-16, using chr() will usually still work correctly for BMP characters.
    # For more complex cases (e.g., surrogate pairs), consider decoding from bytes as UTF-16:
    # byte_data = code_points.tobytes()
    # return byte_data.decode('utf-16-le')
    return ''.join(chr(cp) for cp in code_points)


def hdf5_string_array_to_string_array(dataset, ref_object):
    obj_shape = ref_object.shape
    result = np.empty(obj_shape, dtype=object)
    # Iterate through all dimensions using nested loops
    for idx in np.ndindex(obj_shape):
        # Get the reference at this index
        ref = ref_object[idx]
        # Convert the reference to a string using the helper function
        result[idx] = hdf5_dataset_to_string(dataset[ref])
    return result


def h5_to_dict(h5_obj, dataset, no_attrs=False):
    """
    Recursively convert an HDF5 file or group into a nested dictionary.
    """
    result = edict()
    
    # Extract attributes
    if len(h5_obj.attrs) > 0:
        attrs = {key: h5_obj.attrs[key] for key in h5_obj.attrs}
        if not no_attrs:
            result['__attrs__'] = attrs
    
    # If this is a group, recurse into its members
    if isinstance(h5_obj, h5py.Group):
        for key in h5_obj.keys():
            item = h5_obj[key]
            result[key] = h5_to_dict(item, dataset, no_attrs=no_attrs)
    
    # If this is a dataset, read its data
    elif isinstance(h5_obj, h5py.Dataset):
        data = h5_obj[()]
        # Initialize data_py to None so it's always defined
        data_py = None
        try:
            if 'MATLAB_class' in h5_obj.attrs:
                if h5_obj.attrs['MATLAB_class'] == b'cell':
                    # cell array => object array
                    data_py = np.empty(data.shape, dtype=object)
                    for idx in np.ndindex(data.shape):
                        ref = data[idx]
                        if isinstance(ref, h5py.Reference):
                            data_py[idx] = h5_to_dict(dataset[ref], dataset, no_attrs=no_attrs)
                        else:
                            data_py[idx] = ref
                elif h5_obj.attrs['MATLAB_class'] == b'struct':
                    # TODO maybe to better handle structs?
                    data_py = h5_to_dict(h5_obj, dataset)
                elif h5_obj.attrs['MATLAB_class'] == b'double':
                    data_py = data.astype(np.float64)
                elif h5_obj.attrs['MATLAB_class'] == b'int64':
                    data_py = data.astype(np.int64)
                elif h5_obj.attrs['MATLAB_class'] == b'uint64':
                    data_py = data.astype(np.uint64)
                elif h5_obj.attrs['MATLAB_class'] == b'logical':
                    data_py = data.astype(np.bool_)
                elif h5_obj.attrs['MATLAB_class'] == b'char':
                    try:
                        if isinstance(data, np.ndarray):
                            # Load data fully
                            arr = data[()]
                            # Handle empty char array case
                            if 'MATLAB_empty' in h5_obj.attrs and h5_obj.attrs['MATLAB_empty']:
                                data_py = ''
                            # Handle single character case
                            elif arr.size == 1:
                                # Extract the scalar value and convert to char
                                val = arr.item()  # Convert array of shape (1,1) to a Python scalar
                                data_py = chr(int(val))
                            else:
                                # Normal case: flatten and convert each code point
                                arr_flat = arr.flatten()
                                # Option 1: Direct chr conversion
                                data_py = ''.join(chr(int(cp)) for cp in arr_flat)
                                # Option 2 (if direct chr gives incorrect output, try UTF-16 decoding):
                                # byte_data = arr.astype('<u2').tobytes()
                                # data_py = byte_data.decode('utf-16-le')
                        else:
                            # data is not an ndarray, fallback
                            data_py = str(data)
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Warning: Failed to convert char data: {e}  {h5_obj} {attrs}")
                        data_py = str(data)  # Fallback to string representation
                else:
                    data_py = data  # Default case - use raw data
                    print(f"Warning: Unknown MATLAB class: {h5_obj.attrs['MATLAB_class']} for {h5_obj.name}")
            else:
                # cell array => object array
                data_py = np.empty(data.shape, dtype=object)
                for idx in np.ndindex(data.shape):
                    ref = data[idx]
                    if isinstance(ref, h5py.Reference):
                        data_py[idx] = h5_to_dict(dataset[ref], dataset, no_attrs=no_attrs)
                    else:
                        data_py[idx] = ref
                # No MATLAB class specified - use raw data
                print(f"Warning: No MATLAB class specified for {h5_obj} at path {h5_obj.name}")
        except Exception as e:
            raise ValueError(f"Failed loading path {h5_obj} {h5_obj.name} {attrs}: {str(e)}")
            
        if data_py is None:
            raise ValueError(f"Failed to convert data for {h5_obj} {h5_obj.name}")
            
        result['__data__'] = data_py
    
    return result


def h5_to_dict_simplify(h5_obj, dataset, ):
    """
    Recursively convert an HDF5 file or group into a nested dictionary.
    """
    # if any(h5_obj.name.startswith(path) for path in exluding_paths):
    #     if verbose:
    #         print(f"Excluding {h5_obj.name}")
    #     return None
    # if verbose:
    #     print(f"Loading {h5_obj.name}")
    result = None
    attrs = edict()    # Initialize attrs to empty dict
    
    # Extract attributes
    if len(h5_obj.attrs) > 0:
        attrs = {key: h5_obj.attrs[key] for key in h5_obj.attrs}
    
    # If this is a group, recurse into its members
    if isinstance(h5_obj, h5py.Group):
        result = edict()
        for key in h5_obj.keys():
            item = h5_obj[key]
            result[key] = h5_to_dict_simplify(item, dataset, )
    
    # If this is a dataset, read its data
    elif isinstance(h5_obj, h5py.Dataset):
        data = h5_obj[()]
        data_py = None  # Initialize data_py
        try:
            if 'MATLAB_class' in h5_obj.attrs:
                # if MATLAB_class is set, use it to convert the data
                if h5_obj.attrs['MATLAB_class'] == b'cell':
                    # cell array => object array
                    data_py = np.empty(data.shape, dtype=object)
                    for idx in np.ndindex(data.shape):
                        ref = data[idx]
                        if isinstance(ref, h5py.Reference):
                            data_py[idx] = h5_to_dict_simplify(dataset[ref], dataset, )
                        else:
                            data_py[idx] = ref
                    result = data_py
                elif h5_obj.attrs['MATLAB_class'] == b'struct':
                    # TODO maybe to better handle structs?
                    result = h5_to_dict_simplify(h5_obj, dataset,)
                elif h5_obj.attrs['MATLAB_class'] == b'double':
                    result = data.astype(np.float64)
                elif h5_obj.attrs['MATLAB_class'] == b'single':
                    result = data.astype(np.float32)
                elif h5_obj.attrs['MATLAB_class'] == b'int64':
                    result = data.astype(np.int64)
                elif h5_obj.attrs['MATLAB_class'] == b'int32':
                    result = data.astype(np.int32)
                elif h5_obj.attrs['MATLAB_class'] == b'uint64':
                    result = data.astype(np.uint64)
                elif h5_obj.attrs['MATLAB_class'] == b'logical':
                    result = data.astype(np.bool_)
                elif h5_obj.attrs['MATLAB_class'] == b'char':
                    try:
                        if isinstance(data, np.ndarray):
                            # Load data fully
                            arr = data[()]
                            # Handle empty char array case
                            if 'MATLAB_empty' in h5_obj.attrs and h5_obj.attrs['MATLAB_empty']:
                                data_py = ''
                            # Handle single character case
                            elif arr.size == 1:
                                # Extract the scalar value and convert to char
                                val = arr.item()  # Convert array of shape (1,1) to a Python scalar
                                data_py = chr(int(val))
                            else:
                                # Normal case: flatten and convert each code point
                                arr_flat = arr.flatten()
                                # Option 1: Direct chr conversion
                                data_py = ''.join(chr(int(cp)) for cp in arr_flat)
                                # Option 2 (if direct chr gives incorrect output, try UTF-16 decoding):
                                # byte_data = arr.astype('<u2').tobytes()
                                # data_py = byte_data.decode('utf-16-le')
                        else:
                            # data is not an ndarray, fallback
                            data_py = str(data)
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Warning: Failed to convert char data: {e}  {h5_obj} {attrs}")
                        data_py = str(data)  # Fallback to string representation
                    result = data_py
                else:
                    result = data  # Default case - use raw data
                    print(f"Warning: Existing but unknown MATLAB class: {h5_obj.attrs['MATLAB_class']} for {h5_obj.name}")
            else:
                if h5_obj.dtype == np.float64:
                    result = data.astype(np.float32)
                elif h5_obj.dtype == np.int64:
                    result = data.astype(np.int32)
                elif h5_obj.dtype == np.uint64:
                    result = data.astype(np.uint32)
                elif h5_obj.dtype == np.bool_:
                    result = data.astype(np.bool_)
                else:
                    # No MATLAB class specified - default to cell array => object array 
                    result = np.empty(data.shape, dtype=object)
                    for idx in np.ndindex(data.shape):
                        ref = data[idx]
                        if isinstance(ref, h5py.Reference):
                            result[idx] = h5_to_dict_simplify(dataset[ref], dataset, )
                        else:
                            result[idx] = ref
                    print(f"Warning: No MATLAB class specified for {h5_obj} at path {h5_obj.name} {h5_obj.dtype}")
        except Exception as e:
            raise ValueError(f"Failed loading path {h5_obj} {h5_obj.name} {attrs}: {str(e)}")
            
        if result is None:
            raise ValueError(f"Failed to convert data for {h5_obj} {h5_obj.name}")
            
    return result


# Iterate through the HDF5 dataset and print info about each object
def print_hdf5_info(obj, indent=''):
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {obj.name}, {len(obj.items())} members")
        for key, value in obj.items():
            if obj.name.startswith("/#refs#"):
                continue
            else:
                print_hdf5_info(value, indent + '  ')
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {obj.name}, shape {obj.shape}, type {obj.dtype}")



import time
import pickle as pkl
from os.path import join
from easydict import EasyDict as edict
import h5py
import numpy as np

def load_process_formatted_mat2dict(ephys_name, mat_root, verbose=True):
    """
    Load a single MATLAB file and return its contents as a dictionary.

    Parameters:
        ephys_name (str): The name identifier for the electrophysiological data.
        mat_root (str): The root directory where the .mat files are located.

    Returns:
        dict: A dictionary containing Trials, meta, and rasters data.
    """
    t0 = time.time()
    print(f"Loading {ephys_name} mat file")
    mat_file = join(mat_root, f"{ephys_name}_formatted.mat")
    with h5py.File(mat_file, "r") as data:
        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds for loading mat file")
        # Fast selective loading
        meta_dict = h5_to_dict_simplify(data['meta'], data)
        # Turn rasters into a numpy array, reduce precision to 32-bit float
        rasters = np.array(data['rasters']).astype(np.float32)
        Trial_dict = edict()
        for key in data['Trials'].keys():
            if key in ["B", "MLConfig"]:
                continue
            print(f"Loading {key}", end="\t")
            Trial_dict[key] = h5_to_dict_simplify(data['Trials'][key], data)
            print("done")
        t2 = time.time()
        print(f"Time taken: {t2 - t1:.2f} seconds for loading trials")

    return {
        "Trials": Trial_dict,
        "meta": meta_dict,
        "rasters": rasters,
    }

def batch_process_ephys(ephys_names, mat_root, pkl_root, recompute=False):
    """
    Batch process a list of electrophysiological .mat files and save them as pickle files.

    Parameters:
        ephys_names (list of str): List of electrophysiological data identifiers.
        mat_root (str): The root directory where the .mat files are located.
        pkl_root (str): The directory where the pickle files will be saved.
    """
    for ephys_name in ephys_names:
        if os.path.exists(join(pkl_root, f"{ephys_name}.pkl")) and not recompute:
            print(f"PKL file already exists for {ephys_name}")
            continue
        print(f"Processing {ephys_name}...")
        data_dict = load_process_formatted_mat2dict(ephys_name, mat_root)
        pkl_path = join(pkl_root, f"{ephys_name}.pkl")
        with open(pkl_path, "wb") as f:
            pkl.dump(data_dict, f)
        print(f"Saved {ephys_name} to {pkl_path}\n")
    
    
    