import os
import math
import numpy as np


# Funkcja do wczytywania atrybutów
def read_attr_file(attr_path):
    with open(attr_path, 'r') as f:
        return [line.strip() for line in f]


# Funkcja do wczytywania danych
def read_data_file(data_path):
    with open(data_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]

# Funkcja obliczająca odległość euklidesową między dwoma punktami (na podstawie wspólnych atrybutów)
def euclidean_distance(record1, record2, common_attrs):
    return math.sqrt(sum((float(record1[attr]) - float(record2[attr])) ** 2 for attr in common_attrs))


# Funkcja do uzupełniania brakujących wartości za pomocą k najbliższych sąsiadów
def fill_missing_values(dataset, common_attrs, k=3):
    ds = dataset
    for i, row in enumerate(ds):
        for attr in common_attrs:
            if not row.get(attr):
                # Szukamy k najbliższych sąsiadów, ale tylko na podstawie dostępnych wartości
                distances = []
                for j, other_row in enumerate(dataset):
                    if i == j:
                        continue

                    # Nie ma poszukiwanej wartości w innym wierszu
                    # if attr not in other_row:
                    #     continue

                    # Szukamy wspólnych atrybutów dostępnych w obu wierszach
                    valid_attrs = [
                        a for a in common_attrs
                        if a != attr and a in row and a in other_row
                    ]

                    if not valid_attrs:
                        continue  # Brak atrybutów do porównania

                    dist = euclidean_distance(row, other_row, valid_attrs)
                    distances.append((dist, other_row))

                distances.sort(key=lambda x: x[0])
                nearest = distances[:k]

                # Średnia z dostępnych wartości
                values = [
                    neigh[attr] for _, neigh in nearest
                    if attr in neigh and neigh[attr]
                ]
                
                if values:
                    row[attr] = sum(values) / len(values)
    return ds


folder_to_check = 'male'
base_dir = f'dane/{folder_to_check}'

# Połączymy dane z różnych plików
for subfolder in os.listdir(base_dir)[:1]:
    full_subfolder_path = os.path.join(base_dir, subfolder)

    if not os.path.isdir(full_subfolder_path):
        continue

    # Zbiór, który będziemy uzupełniać
    full_data_list = []

    #Wszystkie atrybuty
    all_attrs = set()
    data_list = list()

    print(f"\n📁 Folder: {subfolder}")
    for i in range(10):  # male-0 do male-9
        prefix = f'{folder_to_check}-{i}'
        attr_path = os.path.join(full_subfolder_path, f'{prefix}.attr')
        data_path = os.path.join(full_subfolder_path, f'{prefix}.data')

        if not os.path.exists(attr_path) or not os.path.exists(data_path):
            continue

        # Wczytanie plików
        attr_names = read_attr_file(attr_path)
        data_rows = read_data_file(data_path)

        #Zbieramy nazwy atrybutów
        all_attrs.update(attr_names)

        # Tworzymy listę słowników
        data_list = [dict(zip(attr_names, row)) for row in data_rows]

        # Łączymy dane z różnych zbiorów
        full_data_list.extend(data_list)

if all_attrs and full_data_list:

    print(f"📦 Połączony zbiór danych zawiera {len(full_data_list)} rekordów i {len(all_attrs)} atrybutów")

    # Przed uzupełnianiem
    print("Przed uzupełnieniem:")
    for row in full_data_list:  # pokaż pierwsze 10 rekordów
        print('   ', row)

    # Uzupełnianie brakujących danych
    filled_dataset = fill_missing_values(full_data_list, all_attrs, k=3)

    # Po uzupełnianiu
    print("\nPo uzupełnieniu:")
    for row in filled_dataset[:10]:  # pokaż pierwsze 10 rekordów po uzupełnieniu
        print('   ', row)
