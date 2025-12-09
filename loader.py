import numpy as np

def load_data(file_path, limit=None):

    print(f"Data loaded from {file_path}")
    file = open(file_path, 'r')
    data = file.readlines()
    file.close()

    data = data[1:]  # Skip header line if present
    
    if limit:
        data = data[:limit]
    
    inputs = []
    targets = []

    for record in data:
        values = record.split(',')
        
        inputs_row = (np.asarray(values[1:], dtype=float)/255.0*0.99) + 0.01
        inputs.append(inputs_row)

        correct_label = int(values[0])

        targets_row = np.zeros(10) + 0.01
        targets_row[correct_label] = 0.99

        targets.append(targets_row)
    
    return np.array(inputs), np.array(targets)

