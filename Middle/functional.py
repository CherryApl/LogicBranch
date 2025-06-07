import torch



def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)

import torch


def bin_op_s_matrix(a, b, i_s, layer):
    # 预定义系数矩阵，确保类型和设备与输入一致
    coeffs1 = torch.tensor([
        [1, -1, 0, 0],  # 12: 1 - a
        [0, 1, 0, 0],   # 3: a
        [0, 0, 1, 0],   # 5: b
        [1, 0, -1, 0]  # 10: 1 - b
    ], dtype=a.dtype, device=a.device)
    
    coeffs2 = torch.tensor([
        [0, 0, 0, 1],   # 1: a * b
        [0, 1, 0, 0]    # 3: a
        # [0, 1, 1, -1] # 7: a + b - ab
    ], dtype=a.dtype, device=a.device)
    
    coeffs3 = torch.tensor([
        [0, 0, 0, 1],   # 1: a * b
        [0, 1, 1, -2], # 6: a + b - 2ab
    ], dtype=a.dtype, device=a.device)
    
    coeffs4 = torch.tensor([
        [1, 0, 0, 0],   # 0: 1
        [0, 0, 0, 1],   # 1: a * b
        [0, 0, 0, 0],   # 0: 0
        [0, 1, 1, -1],  # 7: a + b - ab
    ], dtype=a.dtype, device=a.device)
    
    # 根据layer选择系数矩阵
    if layer == 1:
        coeffs = coeffs1
    elif layer == 2:
        coeffs = coeffs2
    elif layer == 3:
        coeffs = coeffs3
    elif layer == 4:
        coeffs = coeffs4
    else:
        raise ValueError("Invalid layer value. Layer must be 1, 2, 3, or 4.")
    # 计算权重矩阵：i_s与系数矩阵相乘
    weights = torch.matmul(i_s, coeffs)  # 结果形状为[..., 4]
    # 计算基函数a*b
    ab = a * b
    # 组合各基函数的加权和
    result = (
        weights[..., 0] +          # 常数项基（1）的权重
        weights[..., 1] * a +      # a的权重
        weights[..., 2] * b +      # b的权重
        weights[..., 3] * ab       # ab的权重
    )
    return result


# def bin_op_s_matrix(a, b, i_s, layer):
#     # 预定义系数矩阵，确保类型和设备与输入一致
#     coeffs1 = torch.tensor([
#         [1, -1, 0, 0],  # 12: 1 - a
#         [0, 1, 0, 0],   # 3: a
#         [0, 0, 1, 0],   # 5: b
#         [1, 0, -1, 0],  # 10: 1 - b
#     ], dtype=a.dtype, device=a.device)
    
#     coeffs2 = torch.tensor([
#         [0, 0, 0, 1],   # 1: a * b
#         [0, 1, 1, -1],  # 7: a + b - ab
#         [1, -1, -1, 1], # 8: 1 - a - b + ab
#         # [1, 0, 0, -1],  # 14: 1 - ab
#         [0, 0, 0, 0],  # 0: 0
        
#     ], dtype=a.dtype, device=a.device)
    
#     coeffs3 = torch.tensor([
#         [0, 0, 0, 1],   # 1: a * b
#         [0, 1, 1, -1],  # 7: a + b - ab
#         [0, 1, 1, -2],  # 6: a + b - 2ab
#         # [0, 1, 0, 0],   # 3: a
#         [0, 0, 0, 0],  # 0: 0
#     ], dtype=a.dtype, device=a.device)
    
#     coeffs4 = torch.tensor([
#         [1, 0, 0, 0],   # 0: 1
#         [0, 0, 0, 1],   # 1: a * b
#         # [0, 1, 1, -2],  # 6: a + b - 2ab
#         [0, 0, 0, 0],   # 0: 0
#         [0, 1, 1, -1],  # 7: a + b - ab
#     ], dtype=a.dtype, device=a.device)
    
#     # 根据layer选择系数矩阵
#     if layer == 1:
#         coeffs = coeffs1
#     elif layer == 2:
#         coeffs = coeffs2
#     elif layer == 3:
#         coeffs = coeffs3
#     elif layer == 4:
#         coeffs = coeffs4
#     else:
#         raise ValueError("Invalid layer value. Layer must be 1, 2, 3, or 4.")
#     # 计算权重矩阵：i_s与系数矩阵相乘
#     weights = torch.matmul(i_s, coeffs)  # 结果形状为[..., 4]
#     # 计算基函数a*b
#     ab = a * b
#     # 组合各基函数的加权和
#     result = (
#         weights[..., 0] +          # 常数项基（1）的权重
#         weights[..., 1] * a +      # a的权重
#         weights[..., 2] * b +      # b的权重
#         weights[..., 3] * ab       # ab的权重
#     )
#     return result

def bin_op_s(a, b, i_s):
    r = torch.zeros_like(a,device=a.device)
    for i in range(16):
        u = bin_op(a, b, i)
        r = r + i_s[..., i] * u
    return r

def conv_cal(seq_factor,seq_const,x,num_subs):
    assert seq_const.shape == seq_factor.shape, 'seq_2_factor and seq_2_const must have same shape'
    assert (x.shape[-1] == seq_factor.shape[-1])&(x.shape[-2] == seq_factor.shape[-2]), 'x and seq_2_factor must have same last two dimension'
    
    res = seq_const[:,0] + x[...,0] * seq_factor[:,0]
    for col in range(1,num_subs):
        cur = seq_const[:,col] + x[...,col] * seq_factor[:,col]
        res = res * cur
    return res