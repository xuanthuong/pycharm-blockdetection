import os
from img_pro_helper import cut_block_queue
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print (os.path.join(BASE_DIR, '.output'))
cut_block_queue(os.path.join(BASE_DIR, 'input', 'testSI.jpg'), os.path.join(BASE_DIR, '.output\\'))
