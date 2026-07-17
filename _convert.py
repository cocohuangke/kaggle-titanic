import json, os

with open(r'\\DS1019\home\Drive\project\titanic\titanic-v7.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = []
lines.append('# Auto-generated from titanic-v7.ipynb\n')
lines.append('import sys, os\n')
lines.append('os.chdir("//DS1019/home/Drive/project/titanic")\n')
lines.append('\n')

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        lines.append(f'# ====== CELL {i} ======\n')
        for line in cell['source']:
            lines.append(line)
        lines.append('\n')

with open(r'\\DS1019\home\Drive\project\titanic\titanic-v7-run.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Wrote {len(lines)} lines')
