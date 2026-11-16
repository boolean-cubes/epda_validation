"""Parses a provided .hdf5 file from the HamLib library and converts the Hamiltonian (decomposed as a Pauli string with complex coefficients) into a matrix using Qiskit."""

import h5py
import sys
import re
from qiskit.quantum_info import SparsePauliOp


def parse_hamlib_text(hamlib_text, num_qubits):
    """Parses bracketed Hamlib text into a Qiskit SparsePauliOp."""
    pauli_list = []
    
    # Regex to capture the number inside () and the operators inside []
    pattern = r"\(([^)]+)\)\s*\[([^\]]*)\]"
    
    # finditer loops through every match in the giant text block
    for match in re.finditer(pattern, hamlib_text):
        coef_str = match.group(1)  # e.g., "6.373204203751832+0j"
        ops_str = match.group(2)   # e.g., "X0 X1 X2 Y3 Y5" or ""
        
        # Convert string to a complex number
        coef = complex(coef_str)
        
        # Start with an array of Identities
        pauli_array = ['I'] * num_qubits
        
        # If the brackets weren't empty, fill in the operations
        if ops_str.strip():
            ops = ops_str.strip().split()
            for op in ops:
                letter = op[0]      # 'X', 'Y', or 'Z'
                idx = int(op[1:])   # Qubit index (e.g., 0, 12, 19)
                pauli_array[idx] = letter
                
        # Reverse for Qiskit's right-to-left endianness
        qiskit_string = "".join(pauli_array)[::-1]
        pauli_list.append((qiskit_string, coef))
        
    return SparsePauliOp.from_list(pauli_list)

# --- Main Execution ---
if len(sys.argv) < 2:
    print("Usage: python3 decomp_to_matrix.py <filename.hdf5>")
    sys.exit(1)

# 2. Grab the filename from the first command-line argument
filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
    first_instance = list(f.keys())[0]
    dataset = f[first_instance]
    
    # 1. Grab the number of qubits from the dataset's attributes
    num_qubits = int(dataset.attrs['nqubits'])
    
    # 2. Extract the scalar dataset
    raw_data = dataset[()]
    
    # 3. Decode if it is stored as bytes
    if isinstance(raw_data, bytes):
        text_data = raw_data.decode('utf-8')
    else:
        text_data = raw_data

# 4. Parse the text into a Qiskit Operator
operator = parse_hamlib_text(text_data, num_qubits)

print(f"--- Loaded {first_instance} ---")
print(f"Qubits: {operator.num_qubits}")
print(f"Total Terms: {len(operator)}")

# 5. Convert to Matrix
if operator.num_qubits > 12:
    print("\nConverting to Sparse Matrix (Dense matrix would be too large)...")
    matrix = operator.to_matrix(sparse=True)
else:
    print("\nConverting to Dense Matrix...")
    matrix = operator.to_matrix()

print(matrix)

#print("\n--Full Operator--")
# print(operator)

print("\n--- Pauli Strings and Coefficients ---")
for pauli, coeff in zip(operator.paulis, operator.coeffs):
    # pauli is a Pauli object, we convert it to a string with .to_label()
    print(f"[{pauli.to_label()}] : {coeff}")
