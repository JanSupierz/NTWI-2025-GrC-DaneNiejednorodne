import os
import re

base_root = 'dane'  # Główny katalog, zawierający "male", "duze", itd.

# Automatycznie znajdujemy wszystkie podfoldery w 'dane/'
base_folders = [
    os.path.join(base_root, name)
    for name in os.listdir(base_root)
    if os.path.isdir(os.path.join(base_root, name))
]

for base in base_folders:
    prefix = os.path.basename(base)  # np. "male" lub "duze"

    for subfolder in os.listdir(base):
        folder_path = os.path.join(base, subfolder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            match = re.match(rf'^{re.escape(prefix)}(\d+)\.(attr|data)$', filename)
            if match:
                number, ext = match.groups()
                new_name = f"{prefix}-{number}.{ext}"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)

                if filename != new_name:
                    print(f"{old_path} ➝ {new_path}")
                    os.rename(old_path, new_path)
