import os

# print(os.getcwd())

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

# Example usage:
directories_to_create = [
    'examples/penalties/plots',
    'examples/trees/plots',
    'examples/trees/plots/noerrxy',
    'examples/trees/plots/noerrxy/one_eps',
    'examples/trees/plots/noerrxy12',
    'examples/trees/plots/noerrxy12/one_eps',
    'examples/trees/plots/errxy_1IP',
    'examples/trees/plots/errxy_1IP/one_eps',
    'examples/trees/plots/errxy_2IPS',
    'examples/trees/plots/errxy_2IPS/one_eps',
    'examples/trees/plots/errxy12',
    'examples/trees/plots/errxy12/one_eps'
]

create_directories(directories_to_create)
