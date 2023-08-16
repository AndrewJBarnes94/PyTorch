import torch

print(torch.__version__)

### Creating tensors with torch.tensor()
scalar = torch.tensor(7)            # Scalar tensor

print(scalar)
print(scalar.item)                  # Get tensor back as Python int

vector = torch.tensor([7, 7])       # Vector tensor

print(vector)
print(vector.item)

print(vector.ndim)                  # Tells the amount of dimensions (number of brackets)
print(vector.shape)

MATRIX = torch.tensor([
    [7, 8],
    [9, 10]
])

print(MATRIX.ndim)
print(MATRIX.shape)

TENSOR = torch.tensor([
    [
        [1, 2, 3],
        [3, 6, 9],
        [2, 4, 6]
    ]
])

print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

### Random Tensors

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)

print(random_tensor)
print(random_tensor.ndim)
print(random_tensor.shape)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3))   # height, width, color channels (R, G, B)

print(random_image_size_tensor)
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

### Zeros and Ones

# Create a tensor of all zeros
zero = torch.zeros(size=(3, 4))

print(zero)

# Create a tnesor of all ones

ones = torch.ones(size=(3, 4))

print(ones)
print(ones.dtype)                       # Shows float32

### Create a range of tensors and use tensor-like functionality

# Use torch.range() or torch.arange() if torch.range() is depreciated
one_to_ten = torch.arange(1, 11)
print(one_to_ten)

one_to_100_with_steps = torch.arange(start=1, end=100, step=20)
print(one_to_100_with_steps)

# Using a pre-existing tensor to create the shape of a new tensor
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)


### Tensor Datatypes

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None)
print(float_32_tensor.dtype)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)
print(float_16_tensor.dtype)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,              # What the datatype is
                               device=None,             # default is "cpu", but could be cuda
                               requires_grad=False)     # Whether or not to track gradients with this tensors operation

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

print(float_16_tensor * float_32_tensor)                # Surprisingly doesn't error out

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(float_16_tensor * int_32_tensor)

long_32_tensor = torch.tensor([3, 6, 9], dtype=torch.long)
print(long_32_tensor)

print(float_32_tensor * long_32_tensor)

### Manipulating Tensors

# Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

# Multiply tensor by 10
print(tensor * 10)

# Subtract 10
print(tensor - 10)


### Matrix Multiplication

# Element wise multiplication
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")

# Matrix multiplication
print(torch.matmul(tensor, tensor))

# Matrix multiplication by hand
print(1*1 + 2*2 + 3*3)

value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value)


print(torch.matmul(torch.rand(3,2), torch.rand(2,5)))


tensor_a = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6],
])

tensor_b = torch.tensor([
    [7, 10],
    [8, 11],
    [9, 12],
])

# Can't do: print(torch.matmul(tensor_a, tensor_b))
# Because there is a shape error

# Transpose tensor_b and compare to tensor_a
print(tensor_b.T.shape)
print(tensor_a.shape)

print(torch.matmul(tensor_a, tensor_b.T))


### Finding the min, max, mean, sum, etc (tensor aggregation)

# Create a tensor
x = torch.arange(0, 100, 10)
print(x)

# Find the min
print(torch.min(x))
print(x.min())

# Find the max
print(torch.max(x))
print(x.max())

# Find the mean ~ note: torch.mean() requires a tensor of float32 to work
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())

# Find the sum
print(torch.sum(x))
print(x.sum())


### Find thepositional min and max

# Find the position in tensor that has the minimum value with argmin() -> returns index 
print(x.argmin())
print(x[0])

# Find the position in tensor that has the maximum value with argmax()
print(x.argmax())
print(x[9])


### Reshaping, stacking, squeezing, and unsqueezing tensors

x = torch.arange(1., 11.)
print(x)
print(x.shape)

x_reshape = x.reshape(5, 2)
print(x_reshape)
print(x_reshape.shape)

x = torch.arange(1., 10.)
print(x)
print(x.shape)

# Add an extra dimension
x_reshape = x.reshape(1, 9)
print(x_reshape)
print(x_reshape.shape)

# Change the view (shares the same memory as x)
z = x.view(1, 9)
print(z.shape)

# Changing z changes x because of same memory
z[:, 0] = 5
print(z)
print(x)

# Stack tensors on top of eachother
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor: {x_reshape}")
print(f"Previous shape: {x_reshape.shape}")

# Remove extra dimensions from x_reshaped
x_squeezed = x_reshape.squeeze()
print(f"\nNew tensor: {x_reshape.squeeze().shape}")
print(f"New shape: {x_squeezed.shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# torch.permute - rearranges the dimesions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # height, width, color channels

# Permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # Shifts axis 0 to 1, 1 to 2, and 2 to 0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")

### Indexing (selecting data from tensors)

# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)

# Index on new tensor
print(x[0])
print(x[0, 0])
print(x[0] [0]) # same as above
print(x[0][0][0])

# Get all values of 0th and 1st dimensions, but only index 1 of 2nd dimension
print(x[:, :, 1])

# Get all values of the 0th dimensions, but only the 1 index value of 1st and 2nd dimension
print(x[:, 1, 1])

# Get index 0 of 0th and first dimension and all values of 2nd dimension
print(x[0, 0, :])

# Index on x to return 9
print(x[0][2][2])

# Index on x to return 3, 6, 9
print(x[:, :, 2])