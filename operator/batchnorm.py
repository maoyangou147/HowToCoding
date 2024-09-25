import numpy as np

# def batchnorm_forward(x, gamma, beta, bn_param):
#     mode = bn_param['mode']
#     eps = bn_param.get('eps', 1e-5)
#     momentum = bn_param.get('momentum', 0.9)

#     N, D = x.shape
#     running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
#     running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

#     out, cache = None, None

#     if mode == 'train':
#         sample_mean = np.mean(x, axis=0)
#         sample_var = np.var(x, axis=0)

#         x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
#         out = gamma * x_hat + beta

#         cache = (gamma, x, sample_mean, sample_var, eps, x_hat)

#         running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#         running_var = momentum * running_var + (1 - momentum) * sample_var
#     elif mode == 'test':
#         # x_hat = (x - running_mean) / np.sqrt(sample_var + eps)
#         # out = gamma * x_hat + beta
#         scale = gamma / (np.sqrt(running_var + eps))
#         out = x * scale + (beta - running_mean * scale)
#     else:
#         raise ValueError('Error')
    
#     bn_param['running_mean'] = running_mean
#     bn_param['running_var'] = running_var

#     return out, cache

def batchnorm_forward(x, gamma, beta, bn_params):
    # x:(B, N)

    mode = bn_params['mode']
    eps = bn_params.get('eps', 1e-5)
    momentum = bn_params.get('momentum', 0.9)

    B, N = x.shape
    running_mean = bn_params.get('running_mean', np.zeros(N, dtype=x.dtype))
    running_var = bn_params.get('running_var', np.zeros(N, dtype=x.dtype))

    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        output = gamma * x_hat + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'val':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        output = gamma * x_hat + beta
    else:
        raise ValueError('Error')
    
    bn_params['running_mean'] = running_mean
    bn_params['running_var'] = running_var

    return output




