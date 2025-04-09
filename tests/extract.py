import json

def extract_code_from_ipynb(ipynb_file):
    with open(ipynb_file) as f:
        data = json.load(f)
    code_cells = [cell for cell in data['cells'] if cell['cell_type'] == 'code']
    code = '\n'.join([''.join(cell['source'])  for cell in code_cells if cell['source'][0].startswith('class')] )
    return code

# Replace "example" with YOUR NAME
# name = 'YOUR NAME'
name = 'lhz'
code = extract_code_from_ipynb(f'backpropogation_{name}.ipynb')
with open(f'full_{name}.py','w') as f:
    f.write("""import numpy as np\n
from module import Zeros, XavierUniform, Activation, module\n\n""")
    f.write(code)