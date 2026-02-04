import json

with open('previous_train_model.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'class ConditionalGAN' in source or 'create_generator_model' in source or 'create_discriminator_model' in source:
            print("--- CELL START ---")
            print(source)
            print("--- CELL END ---")
