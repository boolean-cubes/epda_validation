import numpy as np

def reconstruct_dense_matrix(coefficients_dict: dict[tuple[int, int], complex], n: int) -> np.ndarray:
    """
    Reconstructs a 2^n x 2^n dense matrix directly from the output 
    dictionary of `pauli_decompose`.
    """
    dim = 1 << n  # equivalent to 2^n
    M = np.zeros((dim, dim), dtype=complex)
    
    # Helper functions mirroring the repository's math logic
    def i_power(k: int) -> complex:
        return (1, 1j, -1, -1j)[k & 3]
        
    def parity(val: int) -> int:
        return val.bit_count() & 1

    # Iterate through all u (columns)
    for u in range(dim):
        for (x, z), alpha in coefficients_dict.items():
            
            # --- ROW CALCULATION (Lemma 1) ---
            # Row index v is calculated via native bitwise XOR
            v = x ^ u
            
            # --- PHASE CALCULATION (Lemma 2) ---
            # i^{|x ∧ z|} -> computed using bitwise AND and bit_count()
            overlap_phase = i_power((x & z).bit_count())
            
            # (-1)^{z · u} -> computed using bitwise AND and parity
            z_dot_u_sign = -1 if parity(z & u) else 1
            
            # Combine the phases
            matrix_element = overlap_phase * z_dot_u_sign
            
            # Add the weighted contribution to the matrix
            M[v, u] += alpha * matrix_element
            
    return M
