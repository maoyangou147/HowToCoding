import numpy as np

def conv2d_numpy(input_data, kernel, stride=1, padding=0):
    input_channels, input_height, input_width = input_data.shape
    n_kernels, _, kernel_height, kernel_width = kernel.shape

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    output_data = np.zeros((n_kernels, output_height, output_width), dtype=np.float32)

    if padding > 0:
        input_data = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    for k in range(n_kernels):
        for i in range(0, input_height - kernel_height + 1, stride):
            for j in range(0, input_width - kernel_width + 1, stride):
                output_data[k, i // stride, j // stride] = np.sum(input_data[:, i:i+kernel_height, j:j+kernel_width] * kernel[k])
    
    return output_data



image = np.random.randn(3,28,28)
kernel = np.random.randn(2,3,3,3)

result = conv2d_numpy(image, kernel, stride=2, padding=1)


print(result.shape)
