import json

def count_lines_in_jupyter_notebook(notebook_path):
    total_lines = 0
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = json.load(f)

    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            total_lines += len(cell['source'])
    return total_lines

notebook_file = 'analysis.ipynb'
lines = count_lines_in_jupyter_notebook(notebook_file)
print(f"Total lines of code in '{notebook_file}': {lines}")
