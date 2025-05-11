import os

def read_attr_file(attr_path):
    with open(attr_path, 'r') as f:
        return [line.strip() for line in f]

def read_data_file(data_path):
    with open(data_path, 'r') as f:
        return [line.strip().split() for line in f]


checkfolder = 'male'
base_dir = f'dane/{checkfolder}'

for subfolder in os.listdir(base_dir)[:1]:
    full_subfolder_path = os.path.join(base_dir, subfolder)

    if not os.path.isdir(full_subfolder_path):
        continue

    print(f"\n📁 Folder: {subfolder}")
    for i in range(10):  # male-0 do male-9
        prefix = f'{checkfolder}-{i}'
        attr_path = os.path.join(full_subfolder_path, f'{prefix}.attr')
        data_path = os.path.join(full_subfolder_path, f'{prefix}.data')

        if not os.path.exists(attr_path) or not os.path.exists(data_path):
            continue

        attr_names = read_attr_file(attr_path)
        data_rows = read_data_file(data_path)
        dataset = [dict(zip(attr_names, row)) for row in data_rows]

        print(f'📦 Zbiór {prefix}: {len(dataset)} rekordów')
        for row in dataset:
            print('   ', row)


