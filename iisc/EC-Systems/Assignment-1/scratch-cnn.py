import random

# Used for math.exp in sigmoid
import math

# For command-line argument parsing
import argparse

# Constants for the CNN
NUM_INPUT_FMAPS = 8
NUM_INPUT_CHANNELS = 4
INPUT_FMAP_HEIGHT = 28
INPUT_FMAP_WIDTH = 28
KERNEL_HEIGHT = 5
KERNEL_WIDTH = 5
NUM_KERNELS = 16

# Assuming POOL_SIZE of 2x2 for Average Pooling
POOL_SIZE = 2
POOL_STRIDE = 2 # For conducting non-overlapping pooling

# Parameters based on question
PADDING = 0
STRIDE = 2

# Calculate Output Fmap Dimensions from Input Fmap and Kernel
NUM_OUTPUT_FMAPS = NUM_INPUT_FMAPS
NUM_OUTPUT_CHANNELS = NUM_KERNELS
OUTPUT_FMAP_HEIGHT = ((INPUT_FMAP_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) // STRIDE) + 1
OUTPUT_FMAP_WIDTH = ((INPUT_FMAP_WIDTH - KERNEL_WIDTH + 2 * PADDING) // STRIDE) + 1

# Initialize 4D Array Variables
INPUT_FMAP = []
KERNEL = []
OUTPUT_FMAP = []

# Function to generate a random signed float between -1.0 and 1.0
def get_random_signed_float():
    rand_value = random.uniform(-1.0, 1.0)
    return rand_value

# Sigmoid function to bring non-linearity after convolution to capture complex features
def sigmoid(x):
    return 1 / (1 + math.exp((-1) * x))

# Function to populate input fmap and kernel with random values
def initialize_4d_array(num_objects, num_channels, height, width, randomise: bool = True) -> list:
    # We use python short-hand notation here to create a 4D array
    array_4d = [
        [
         [
             [  # To use the same function to initialize any 4D array with 0s, we use a boolean variable randomise
                get_random_signed_float() if randomise else 0.0 for _ in range(width) 
             ]  for _ in range(height)
         ]   for _ in range(num_channels)
        ] for _ in range(num_objects)
    ]
    
    return array_4d

# Helper function to print the 4D array (for verification)
def print_4d_array(array_4d, name):
    print(f"\n{'='*80}")
    print(f"{name} - Shape: ({len(array_4d)}, {len(array_4d[0])}, {len(array_4d[0][0])}, {len(array_4d[0][0][0])})")
    print(f"{'='*80}\n")
    
    for i in range(len(array_4d)):
        print(f"Feature Map {i}:")
        for j in range(len(array_4d[i])):
            print(f"  Channel {j}:")
            for k in range(len(array_4d[i][j])):
                print(f"    ", end="")
                for l in range(len(array_4d[i][j][k])):
                    print(f"{array_4d[i][j][k][l]:8.4f}", end=" ")
                print()
            print()
        print(f"{'-'*80}\n")

# Helper function to transpose a 2D matrix (Used for toeplitz matrix)
def transpose_2d_matrix(matrix):
    transposed = []
    for col in range(len(matrix[0])):
        new_row = []
        for row in range(len(matrix)):
            new_row.append(matrix[row][col])
        transposed.append(new_row)
    return transposed

# Helper function to multiply two 2D matrices (Used for toeplitz CNN operation)
def matrix_multiply_2d(A, B):
    result = []
    
    # Traverse through each row of A
    for i in range(len(A)):
        result_row = []
        
        # Traverse through each column of B
        for j in range(len(B[0])):
            sum = 0.0
            
            # Calculate dot product between row vector from A and column vector from B
            for k in range(len(B)):
                sum += A[i][k] * B[k][j]
            result_row.append(sum)
        result.append(result_row)
    return result

# Helper function to compare two 4D arrays for equality (for final verification). We take a tolerance value for float comparison
def compare_4d_arrays(array1, array2, tolerance=1e-5):
    if len(array1) != len(array2):
        return False
    for i in range(len(array1)):
        if len(array1[i]) != len(array2[i]):
            return False
        for j in range(len(array1[i])):
            if len(array1[i][j]) != len(array2[i][j]):
                return False
            for k in range(len(array1[i][j])):
                if len(array1[i][j][k]) != len(array2[i][j][k]):
                    return False
                for l in range(len(array1[i][j][k])):
                    if abs(array1[i][j][k][l] - array2[i][j][k][l]) > tolerance:
                        return False
    return True

# Function to conduct average pooling across any feature map
def avg_pooling(FEATURE_MAP):
    NUM_MAPS = len(FEATURE_MAP)
    NUM_CHANNELS = len(FEATURE_MAP[0])
    ORIG_HEIGHT = len(FEATURE_MAP[0][0])
    ORIG_WIDTH = len(FEATURE_MAP[0][0][0])
    
    # Calculate new dimensions based on pooling parameters
    NEW_HEIGHT = ((ORIG_HEIGHT - POOL_SIZE + 2 * PADDING) // POOL_STRIDE) + 1
    NEW_WIDTH = ((ORIG_WIDTH - POOL_SIZE + 2 * PADDING) // POOL_STRIDE) + 1
    
    # Get a new 4D Array with 0 values
    NEW_FMAP = initialize_4d_array(NUM_MAPS, NUM_CHANNELS, NEW_HEIGHT, NEW_WIDTH, randomise=False)
    
    # First 4 loops iterate over each pixel in the final output fmap
    for fmap_i in range(NUM_MAPS):
        for channel_i in range(NUM_CHANNELS):
            for height_i in range(NEW_HEIGHT):
                for width_i in range(NEW_WIDTH):
                    
                    # To store sum of pixels under the pooling window
                    sum = 0.0
                    
                    # Loop over the pooling window
                    for pool_height_i in range(POOL_SIZE):
                        for pool_width_i in range(POOL_SIZE):
                            
                            # Calculate offsets in the input image for the pooling window and sum the values
                            fmap_height_offset = height_i * POOL_STRIDE + pool_height_i - PADDING
                            fmap_width_offset = width_i * POOL_STRIDE + pool_width_i - PADDING
                            
                            # If PADDING > 0, the offsets can go out of bounds. In this case, we append a 0
                            if 0 <= fmap_height_offset < ORIG_HEIGHT and 0 <= fmap_width_offset < ORIG_WIDTH:
                                pixel = FEATURE_MAP[fmap_i][channel_i][fmap_height_offset][fmap_width_offset]
                            else:
                                pixel = 0
                            
                            sum += pixel
                    
                    # Update the new, reduced feature map with the average of all pixels under the pooling window   
                    NEW_FMAP[fmap_i][channel_i][height_i][width_i] = sum / (POOL_SIZE * POOL_SIZE)
                    
    return NEW_FMAP

# Function to perform the convolution using Naive Loop approach
def naive_loop_cnn(INPUT_FMAP, KERNEL):
    # First initialize an empty Output Feature Map
    OUTPUT_FMAP = initialize_4d_array(NUM_OUTPUT_FMAPS, NUM_OUTPUT_CHANNELS, OUTPUT_FMAP_HEIGHT, OUTPUT_FMAP_WIDTH, randomise=False)
    
    # First four loops iterate over each pixel in the output feature map
    for ofmap_i in range(NUM_OUTPUT_FMAPS):
        for out_channel_i in range(NUM_OUTPUT_CHANNELS):
            for out_height_i in range(OUTPUT_FMAP_HEIGHT):
                for out_width_i in range(OUTPUT_FMAP_WIDTH):
                    
                    # Initialize sum for the convolution operation for current output pixel
                    sum = 0.0
                    
                    # Next 3 loops iterate over each kernel for all the channels
                    for in_channel_i in range(NUM_INPUT_CHANNELS):
                        for kernel_height_i in range(KERNEL_HEIGHT):
                            for kernel_width_i in range(KERNEL_WIDTH):
                                
                                # We calculate the pixel offset in the Input Fmap based on the current active Output Fmap pixel 
                                ifmap_height_offset = out_height_i * STRIDE + kernel_height_i - PADDING
                                ifmap_width_offset = out_width_i * STRIDE + kernel_width_i - PADDING
                                
                                # If PADDING > 0, the offsets can go out of bounds. In this case, we append a 0
                                if 0 <= ifmap_height_offset < INPUT_FMAP_HEIGHT and 0 <= ifmap_width_offset < INPUT_FMAP_WIDTH:
                                    pixel = INPUT_FMAP[ofmap_i][in_channel_i][ifmap_height_offset][ifmap_width_offset]
                                else:
                                    pixel = 0
                                
                                # Sum is updated for each multiplication between a kernel pixel and Input Fmap pixel for the corresponding Output fmap
                                sum += pixel * KERNEL[out_channel_i][in_channel_i][kernel_height_i][kernel_width_i]
                                
                    # Output Fmap pixel is finally updated with the sigmoid of the sum of products of all channels from a single Kernel and Input Fmap to get a single output channel
                    # As we iterate over the output channels, all other kernels are multiplied with all Input Fmaps to get the rest of the output channels
                    OUTPUT_FMAP[ofmap_i][out_channel_i][out_height_i][out_width_i] = sigmoid(sum)

    # Perform Average Pooling on the generated Output Fmap
    POOLED_OUTPUT_FMAP = avg_pooling(OUTPUT_FMAP)
    return POOLED_OUTPUT_FMAP

# Function to generate the flattened kernel
def kernel_to_toeplitz_matrix(KERNEL):
    toeplitz_matrix = []
    
    # Iterate over each kernel
    for kernel_i in range(len(KERNEL)):
        
        # Create a 1D array for all channels within a kernel
        channel_1D = []
        
        # Loop over each row in each channel and flatten the contents of each row into the 1D array
        for channel_i in range(len(KERNEL[kernel_i])):
            for height_i in range(len(KERNEL[kernel_i][channel_i])):
                channel_1D.extend(KERNEL[kernel_i][channel_i][height_i])
        
        # Finally, append the 1D array of all channels to the 2D Toeplitz Matrix
        toeplitz_matrix.append(channel_1D)
        
        # Print dimensions for verification
    # print(f"\nToeplitz Kernel Matrix - Shape: ({len(toeplitz_matrix)}, {len(toeplitz_matrix[0])})\n")
    return toeplitz_matrix

# Function to generate the flattened feature maps
def fmap_to_toeplitz_matrix(INPUT_FMAP, KERNEL):
    toeplitz_matrix = []
    
    # Calculate necessary dimensions locally to increase function reusability
    NUM_INPUT_FMAPS = len(INPUT_FMAP)
    NUM_INPUT_CHANNELS = len(INPUT_FMAP[0])
    KERNEL_HEIGHT = len(KERNEL[0][0])
    KERNEL_WIDTH = len(KERNEL[0][0][0])
    OUTPUT_FMAP_HEIGHT = ((len(INPUT_FMAP[0][0]) - KERNEL_HEIGHT + 2 * PADDING) // STRIDE) + 1
    OUTPUT_FMAP_WIDTH = ((len(INPUT_FMAP[0][0][0]) - KERNEL_WIDTH + 2 * PADDING) // STRIDE) + 1
    
    # First three loops to iterate over the output fmap dimensions
    # Although we don't calculate the output feature map here, the traversing of the sliding window will be done using the Output FMAP dimensions as reference 
    for fmap_i in range(NUM_INPUT_FMAPS):
        for height_i in range(OUTPUT_FMAP_HEIGHT):
            for width_i in range(OUTPUT_FMAP_WIDTH):
                
                # A 1D array to store flattened contents for one patch from all the channels 
                kernel_patch = []
                
                # Looping over kernel dimensions to get each pixel under sliding window for each channel
                for channel_i in range(NUM_INPUT_CHANNELS):
                    for kernel_height_i in range(KERNEL_HEIGHT):
                        for kernel_width_i in range(KERNEL_WIDTH):
                            
                            # Calculating offsets for the sliding window based on the output fmap dimension
                            fmap_height_offset = height_i * STRIDE + kernel_height_i - PADDING
                            fmap_width_offset = width_i * STRIDE + kernel_width_i - PADDING
                            
                            # If PADDING > 0, the offsets can go out of bounds. In this case, we append a 0
                            if 0 <= fmap_height_offset < INPUT_FMAP_HEIGHT and 0 <= fmap_width_offset < INPUT_FMAP_WIDTH:
                                pixel = INPUT_FMAP[fmap_i][channel_i][fmap_height_offset][fmap_width_offset]
                            else:
                                pixel = 0
                            
                            # Append each element under the sliding window to the 1D array
                            kernel_patch.append(pixel)
                        
                # Append the 1D Array to the Toeplitz Matrix
                toeplitz_matrix.append(kernel_patch)
    
    # Final array after loops have dimensions [NUM_INPUT_FMAPS * OUTPUT_FMAP_HEIGHT * OUTPUT_FMAP_WIDTH][KERNEL_HEIGHT * KERNEL_WIDTH * NUM_INPUT_CHANNELS]
    # We transpose the matrix to get the desired shape for multiplication later
    toeplitz_matrix = transpose_2d_matrix(toeplitz_matrix)                
    # print(f"\nToeplitz Matrix - Shape: ({len(toeplitz_matrix)}, {len(toeplitz_matrix[0])})\n")
    return toeplitz_matrix

def toeplitz_cnn(INPUT_FMAP, KERNEL):
    # Get flattened Kernel
    TOEPLITZ_KERNEL_MATRIX = kernel_to_toeplitz_matrix(KERNEL)
    
    # Get flattened Input FMAP, with correct dimensions for multiplication
    TOEPLITZ_IFMAP_MATRIX = fmap_to_toeplitz_matrix(INPUT_FMAP, KERNEL)
    
    # Get the final Output 2D Matrix with the dimensions [NUM_KERNELS][NUM_INPUT_FMAPS * OUTPUT_FMAP_HEIGHT * OUTPUT_FMAP_WIDTH]
    OUTPUT_FMAP_2D = matrix_multiply_2d(TOEPLITZ_KERNEL_MATRIX, TOEPLITZ_IFMAP_MATRIX)
    
    # Now we need to reshape the 2D Output Fmap back to 4D, required for pooling and comparing with Naive Loop output
    
    # Initialize empty 4D Output Fmap
    OUTPUT_FMAP_4D = initialize_4d_array(NUM_OUTPUT_FMAPS, NUM_OUTPUT_CHANNELS, OUTPUT_FMAP_HEIGHT, OUTPUT_FMAP_WIDTH, randomise=False)
    
    # Number of pixels per channel in the output fmap
    pixels_per_ofmap = OUTPUT_FMAP_HEIGHT * OUTPUT_FMAP_WIDTH
    
    # We iterate over each pixel here and map it back to the correct position in the 4D array
    # First loop is along the rows of the 2D output fmap (output channels)
    for out_channel_i in range(len(OUTPUT_FMAP_2D)):
        # Second loop is along the columns of the 2D output fmap (all pixels from all output fmaps for a particular output channel)
        for col_i in range(len(OUTPUT_FMAP_2D[0])):
            # Calculate which output fmap the feature belongs to, and the height and width indices within that fmap
            ofmap_batch_i = col_i // pixels_per_ofmap
            reamining_pixels = col_i % pixels_per_ofmap
            ofmap_height_i = reamining_pixels // OUTPUT_FMAP_WIDTH
            ofmap_width_i = reamining_pixels % OUTPUT_FMAP_WIDTH
            
            # Map the value from the 2D Output Fmap to the corresponding location in the 4D Output FMAP 
            OUTPUT_FMAP_4D[ofmap_batch_i][out_channel_i][ofmap_height_i][ofmap_width_i] = sigmoid(OUTPUT_FMAP_2D[out_channel_i][col_i])
            
    # Perform Average Pooling on the generated Output Fmap
    OUTPUT_FMAP_4D = avg_pooling(OUTPUT_FMAP_4D)
    return OUTPUT_FMAP_4D

# Set up argument parser
parser = argparse.ArgumentParser(description='Run CNN with different implementations')
parser.add_argument('--method', type=str, choices=['naive', 'flattened', 'both'], 
                    default='both', help='Choose CNN implementation: naive, flattened, or both')
args = parser.parse_args()

# Initialize Input Fmap and Kernel with random values
INPUT_FMAP = initialize_4d_array(NUM_INPUT_FMAPS, NUM_INPUT_CHANNELS, INPUT_FMAP_HEIGHT, INPUT_FMAP_WIDTH)
KERNEL = initialize_4d_array(NUM_KERNELS, NUM_INPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH)

# Prettify the output for user understanding

print("\n" + "="*80)
print("CNN Implementation Execution")
print("="*80)

# Run based on selected method
if args.method in ['naive', 'both']:
    print("\n[Running Naive Loop CNN...]")
    OUTPUT_FMAP_NAIVE = naive_loop_cnn(INPUT_FMAP, KERNEL)
    print(f"Naive CNN Output Shape: ({len(OUTPUT_FMAP_NAIVE)}, {len(OUTPUT_FMAP_NAIVE[0])}, "
          f"{len(OUTPUT_FMAP_NAIVE[0][0])}, {len(OUTPUT_FMAP_NAIVE[0][0][0])})")
    print("Naive CNN execution completed.")

if args.method in ['flattened', 'both']:
    print("\n[Running Flattened (Toeplitz) CNN...]")
    OUTPUT_FMAP_FLATTENED = toeplitz_cnn(INPUT_FMAP, KERNEL)
    print(f"Flattened CNN Output Shape: ({len(OUTPUT_FMAP_FLATTENED)}, {len(OUTPUT_FMAP_FLATTENED[0])}, "
          f"{len(OUTPUT_FMAP_FLATTENED[0][0])}, {len(OUTPUT_FMAP_FLATTENED[0][0][0])})")
    print("Flattened CNN execution completed.")

# Compare outputs if both methods were run
if args.method == 'both':
    print("\n" + "-"*80)
    print("Comparing Outputs...")
    print("-"*80)
    are_equal = compare_4d_arrays(OUTPUT_FMAP_NAIVE, OUTPUT_FMAP_FLATTENED)
    print(f"\nAre both CNN outputs equal? {'Yes ✓' if are_equal else 'No ✗'}")

print("\n" + "="*80 + "\n")
