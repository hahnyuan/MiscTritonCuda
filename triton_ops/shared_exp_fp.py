import torch

import triton
import triton.language as tl

"""
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.htm
"""

@triton.jit
def to_shared_fp_kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               nshare: tl.constexpr, 
               expbit: tl.constexpr, 
               manbit: tl.constexpr,
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    # extract exponent and mantissa
    sign_mask = 0x80000000
    exp_mask = 0x7f800000
    mantissa_mask = 0x007FFFFF
    sign_bit = x & sign_mask
    exp = (x & exp_mask) >> 23
    mantissa = x & mantissa_mask
    # find the maximum exponent
    max_exp = tl.max(exp, axis=0)
    delta_exp = max_exp - exp + (23 - manbit)
    # truncate mantissa
    new_mantissa = mantissa >> delta_exp << delta_exp
    # scale the result
    result = tl.zeros_like(x)
    result = result | sign_bit
    result = result | (exp << 23)
    result = result | new_mantissa


    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, result, mask=mask)

def to_shared_fp(x: torch.Tensor, nshare=256,expbit=8,manbit=6):
    # We need to preallocate the output.
    assert x.is_cuda
    n_elements = x.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    x_view_int= x.float().view(torch.int32)
    output = torch.empty_like(x_view_int)
    to_shared_fp_kernel[grid](x_view_int, output, n_elements, nshare, expbit, manbit, BLOCK_SIZE=nshare)
    output_view_back= output.view(torch.float32).to(x.dtype)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output_view_back
