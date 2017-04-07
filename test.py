import os
from img_pro_helper import cut_cell_all_cases
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print (os.path.join(BASE_DIR, '.output'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'non_line_test.tif'), os.path.join(BASE_DIR, '.output\\'))
cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'document65.jpg'), os.path.join(BASE_DIR, '.output\\'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
