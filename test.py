import os
from img_pro_helper import cut_cell_all_cases, find_contours
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print (os.path.join(BASE_DIR, '.output'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'non_line_test.tif'), os.path.join(BASE_DIR, '.output\\'))
cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'WhatsApp.png'), os.path.join(BASE_DIR, '.output\\'))
# cut_cell_all_cases(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
# hough_line(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
# find_contours(os.path.join(BASE_DIR, 'input', 'document74.jpg'), os.path.join(BASE_DIR, '.output\\'))
