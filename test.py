import os
from img_pro_helper import cut_cell_all_cases
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print (os.path.join(BASE_DIR, '.output'))
# cut_block_queue(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
# non_line_cut_block(os.path.join(BASE_DIR, 'input', 'non_line_test.tif'), os.path.join(BASE_DIR, '.output\\'))
cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'document84.jpg'), os.path.join(BASE_DIR, '.output\\'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
