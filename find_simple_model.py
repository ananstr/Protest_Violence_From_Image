import json

with open('Protest_Dectection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
    
for i, cell in enumerate(nb['cells']):
    source_text = ''.join(cell.get('source', []))
    if 'SimpleProtestModel' in source_text:
        print(f'Cell {i+1}: {cell.get("id", "no-id")}')
        print('Source:')
        print(source_text)
        print('='*50)
