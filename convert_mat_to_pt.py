
import scipy.io as sio
import torch
import numpy as np
import os

def parse_unicode_numeric_array(arr):
    
    flat = arr.flatten()
    converted = []

    for s in flat:
        s = str(s).strip()

        # Try complex first (covers floats too)
        try:
            c = complex(s)
            converted.append(c)
            continue
        except Exception:
            pass

        # Try float explicitly
        try:
            f = float(s)
            converted.append(f)
            continue
        except Exception:
            pass

        raise ValueError(f"Cannot convert Unicode numeric string '{s}' to float or complex.")

    out = np.array(converted)
    return out.reshape(arr.shape)


def convert_value(v):
    

    # ----- NUMPY ARRAY -----
    if isinstance(v, np.ndarray):

        # Case: Unicode strings (<Uxx) or object-strings
        if v.dtype.kind in ["U", "S"] or v.dtype == object:
            v = parse_unicode_numeric_array(v)

        # Now v is guaranteed numeric (float or complex)
        if np.iscomplexobj(v):
            return torch.tensor(v, dtype=torch.complex128)
        else:
            return torch.tensor(v, dtype=torch.float64)

    # ----- SCALAR STRING -----
    if isinstance(v, (str, np.str_)):
        arr = parse_unicode_numeric_array(np.array([v], dtype=object))
        if np.iscomplexobj(arr):
            return torch.tensor(arr, dtype=torch.complex128)
        else:
            return torch.tensor(arr, dtype=torch.float64)

    # ----- SCALAR NUMERIC -----
    if isinstance(v, (float, int, complex, np.floating, np.integer, np.complexfloating)):
        if isinstance(v, complex):
            return torch.tensor(v, dtype=torch.complex128)
        else:
            return torch.tensor(v, dtype=torch.float64)

    # Default: leave as-is
    return v


def convert_mat_file(mat_path, pt_path):
    data = sio.loadmat(mat_path)

    out = {}
    skip = {"__header__", "__version__", "__globals__"}

    for k, v in data.items():
        if k in skip:
            continue
        out[k] = convert_value(v)

    torch.save(out, pt_path)
    print(f"Saved: {pt_path}")


files = [
    "disp_copy.mat",
    "res_copy.mat",
    "sim_copy.mat",
    "primary_sidebands.mat"
]

for f in files:
    if os.path.exists(f):
        convert_mat_file(f, f.replace(".mat", ".pt"))
    else:
        print(f"Skipping {f} (not found)")
