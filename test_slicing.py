import numpy as np

# Generate a 2D NumPy array for testing
def generate_array(rows, cols):
    return np.arange(1, rows * cols + 1).reshape(rows, cols)

# Function to test different types of 2D slices with slice objects
def test_2d_slices(arr):
    print("Original Array:")
    print(arr)
    print("\nTesting different 2D slices with slice objects:")

    # Define slice objects
    slices = [
        (slice(1, None), slice(None)),          # Select all rows starting from index 1 (second row) and all columns
        (slice(None), slice(1, 3)),             # Select all rows and columns from index 1 to 2 (second and third columns)
        (slice(None, 2), slice(1, None)),       # Select rows from index 0 to 1 (first two rows) and all columns
        (slice(None, None, 2), slice(None, None, 2))  # Select every second row and every second column
    ]

    for row_slice, col_slice in slices:
        try:
            print(f"\nExecuting: arr[{row_slice}, {col_slice}]")
            result = arr[row_slice, col_slice]
            print(result)
        except Exception as e:
            print(f"Error executing slice: arr[{row_slice}, {col_slice}]")
            print(e)

# Example usage
rows = 4
cols = 5
my_array = generate_array(rows, cols)
test_2d_slices(my_array)
