import os

def read_file_as_binary(file_path):
    with open(file_path, 'rb') as file:
        binary_content = file.read()
    return binary_content

def write_binary_to_file(binary_content, output_file_path):
    with open(output_file_path, 'w') as file:
        for byte in binary_content:
            file.write(f'{byte:08b} ')
    
    print(f'Binary content has been written to {output_file_path}')
    