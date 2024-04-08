import torch
import sys
sys.path.append('.')
import triton
import triton.language as tl
from triton_ops.shared_exp_fp import to_shared_fp


def to_share_exp_fp_raw(data:torch.Tensor, dim:int=-1, nshare=256, expbit=8, manbit=6):
    raw_shape = data.shape
    raw_dtype = data.dtype
    if dim < 0:
        dim += len(raw_shape)

    # pad
    if (raw_shape[dim] % nshare) != 0:
        padsize = nshare - (raw_shape[dim] % nshare)
        padshape = list(raw_shape)
        padshape[dim] = padsize
        paddata = torch.zeros(padshape, dtype=data.dtype, device=data.device)
        data = torch.cat([data, paddata], dim=dim)
    else:
        padsize = 0

    tensor = data.float().view(torch.int32)
    padded_shape = list(tensor.shape)
    newshape = list(tensor.shape)
    newshape[dim] = newshape[dim] // nshare
    newshape.insert(dim+1, nshare)
    tensor = tensor.reshape(*newshape)
    
    int_tensor = tensor.view(torch.int32)
    
    sign_mask = 0x80000000
    exp_mask = 0x7f800000
    mantissa_mask = 0x007FFFFF
    
    sign_bit = int_tensor & sign_mask
    exp = (int_tensor & exp_mask) >> 23
    mantissa = int_tensor & mantissa_mask
    
    if expbit<8:
        exp = (exp-127).clamp(-2**(expbit-1), 2**(expbit-1)-1) + 127

    max_exp = torch.amax(exp, dim+1, keepdim=True)
    delta_exp = max_exp - exp + (23 - manbit)

    # new_mantissa = (mantissa + (2**(rshift.float() - 1)).int()) >> rshift #四舍五入
    new_mantissa = mantissa >> delta_exp << delta_exp #不做四舍五入

    result = torch.zeros_like(int_tensor, dtype=torch.int32)
    result = result | sign_bit
    result = result | (exp << 23)
    result = result | new_mantissa

    # reshape
    if padsize!=0:
        result = torch.split(result.view(padded_shape), raw_shape[dim], dim=dim)[0]
    result = result.reshape(raw_shape).view(raw_dtype)
    return result

if __name__=='__main__':
    torch.manual_seed(0)
    size = [2,8,3]
    x = torch.rand(size, device='cuda')
    output_torch = to_share_exp_fp_raw(x,dim=1,nshare=4)
    output_triton = to_shared_fp(x,dim=1,nshare=4)
    torch.cuda.synchronize()
    print(output_torch)
    print(output_triton)
    print(output_torch-output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    
    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],  # Argument names to use as an x-axis for the plot.
            x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
            line_names=['Triton', 'Torch'],  # Label name for the lines.
            styles=[('blue', '-'), ('green', '-')],  # Line styles.
            ylabel='GB/s',  # Label name for the y-axis.
            plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        ))
    def benchmark(size, provider):
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: to_share_exp_fp_raw(x,nshare=256,manbit=2) , quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: to_shared_fp(x,nshare=256,manbit=2), quantiles=quantiles)
        gbps = lambda ms: 12 * size / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)
    # benchmark.run(print_data=True, show_plots=True)