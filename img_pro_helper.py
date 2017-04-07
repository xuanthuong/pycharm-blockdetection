import cv2
import numpy as np
from collections import deque
import json

LEN_TOL = 20000
LN_TOL = 100000
GAP_TOL = 50
NON_LN_TOL = 50000
LN_SPACE = 35
LN_NONE = 10
WRD_SPACE = 100
WRD_NONE = 1000
MIN_NUM_LN = 3
MIN_LEN_LN = 100000
OFFSET = 3


def line_cut_hrz(queue, result, queue_elm):
    roi_img = queue_elm['roi']
    ln_roi_img = queue_elm['lnRoi']
    (h, w) = roi_img.shape[:2]
    pre_y_pos = 0
    max_len = LEN_TOL * 2
    has_line = False
    for y_pos in range(h):  # y1 -> y2
        tmp = np.sum(ln_roi_img[y_pos:y_pos + 1, 0:w])
        if tmp > max_len:
            max_len = tmp
    for y_pos in range(h):
        if np.sum(ln_roi_img[y_pos:y_pos + 1, 0:w]) > max_len - LEN_TOL:
            if y_pos - pre_y_pos < GAP_TOL:
                continue
            roi = roi_img[pre_y_pos:y_pos + OFFSET, 0:w]
            ln_roi = ln_roi_img[pre_y_pos:y_pos + OFFSET, 0:w]
            queue.append({'roi': roi,
                          'lnRoi': ln_roi,
                          'x1': queue_elm['x1'],
                          'x2': queue_elm['x2'],
                          'y1': pre_y_pos + queue_elm['y1'],
                          'y2': y_pos + queue_elm['y1'],
                          'flag': 'V', })
            pre_y_pos = y_pos
            has_line = True
    if not has_line:
        result.append(queue_elm)
    return queue, result


def line_cut_vrt(queue, result, queue_elm):
    roi_img = queue_elm['roi']
    ln_roi_img = queue_elm['lnRoi']
    (h, w) = roi_img.shape[:2]
    pre_x_pos = 0
    max_len = LEN_TOL * 2
    has_line = False
    for x_pos in range(w):
        tmp = np.sum(ln_roi_img[0:h, x_pos:x_pos + 1])
        if tmp > max_len:
            max_len = tmp
    for x_pos in range(w):
        if np.sum(ln_roi_img[0:h, x_pos:x_pos + 1]) > max_len - LEN_TOL:
            if x_pos - pre_x_pos < GAP_TOL:
                continue
            roi = roi_img[0:h, pre_x_pos:x_pos + 1]
            ln_roi = ln_roi_img[0:h, pre_x_pos:x_pos + 1]
            queue.append({'roi': roi,
                          'lnRoi': ln_roi,
                          'x1': pre_x_pos + queue_elm['x1'],
                          'x2': x_pos + queue_elm['x1'],
                          'y1': queue_elm['y1'],
                          'y2': queue_elm['y2'],
                          'flag': 'H', })
            pre_x_pos = x_pos
            has_line = True
    if not has_line:
        result.append(queue_elm)
    return queue, result


def line_cut_do(queue, result, queue_elm):
    if queue_elm['flag'] == 'H':
        queue, result = line_cut_hrz(queue, result, queue_elm)
    else:  # if queue_elm('flag') == 'V':
        queue, result = line_cut_vrt(queue, result, queue_elm)
    return queue, result


def line_cut_queue(src_img, line_img, out_folder):
    (h, w) = src_img.shape[:2]
    queue = deque([])
    result = deque([])
    queue.append({'roi': src_img,
                  'lnRoi': line_img,
                  'x1': 0,
                  'x2': w,
                  'y1': 0,
                  'y2': h,
                  'flag': 'H', })
    queue, result = line_cut_do(queue, result, queue[0])
    queue.popleft()

    t = 1
    while queue:
        # print('Cutting time: %d' % t)
        queue, result = line_cut_do(queue, result, queue[0])
        queue.popleft()
        t += 1
        break
    # print('Done Cutting Block')
    # print(result)

    i = 1
    for q in queue:
        tmp = q['roi']
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        (h, w) = tmp.shape[:2]
        total_pixels = h * w
        zero_pixels = total_pixels - cv2.countNonZero(tmp)
        # print(zero_pixels)
        if zero_pixels > 100:
            cv2.imwrite(
                out_folder + 'b' + str(i) + '-' + str(q['x1']) + '-' + str(q['x2']) + '-' + str(q['y1']) + '-' + str(
                    q['y2']) + '.jpg', q['roi'])
            # res_obj = {
            #     "x1": q['x1'],
            #     "x2": q['x2'],
            #     "y1": q['y1'],
            #     "y2": q['y2'],
            #     "name": 'b' + str(i) + '-' + str(q['x1']) + '-' + str(q['x2']) + '-' + str(q['y1']) + '-' + str(
            #         q['y2']) + '.jpg'
            # }
            i += 1
            # print(json.dumps(res_obj))


def non_line_cut_hrz(src, gray):
    result = deque([])
    (h, w) = src.shape[:2]
    tmp = 0
    meet_text = False
    for y_pos in range(h):
        if np.sum(gray[y_pos:y_pos + 1, 0:w]) > LN_NONE:
            if not meet_text:
                y1 = y_pos
                # print('y1 = %d' % y1)
                meet_text = True
            tmp = 0
        else:
            tmp += 1

        if tmp > LN_SPACE and meet_text:
            y2 = y_pos
            # print('y2 = %d' % y2)
            roi = src[y1:y2, 0:w]
            ln_roi = gray[y1:y2, 0:w]
            result.append({'roi': roi,
                           'lnRoi': ln_roi,
                           'x1': 0,
                           'x2': w,
                           'y1': y1,
                           'y2': y2,
                           'flag': 'NL', })
            tmp = -9999
            meet_text = False
    return result


def non_line_cut_vrt(queue):
    result = deque([])
    for q in queue:
        src_roi = q['roi']
        gray_roi = q['lnRoi']
        (h, w) = src_roi.shape[:2]
        tmp = 0
        meet_text = False
        for x_pos in range(w):
            if np.sum(gray_roi[0:h, x_pos:x_pos + 1]) > WRD_NONE:
                if not meet_text:
                    x1 = x_pos
                    # print('x1 = %d' % x1)
                    meet_text = True
                tmp = 0
            else:
                tmp += 1
            if tmp > WRD_SPACE and meet_text:
                x2 = x_pos
                # print('x2 = %d' % x2)
                roi = src_roi[0:h, x1:x2]
                ln_roi = gray_roi[0:h, x1:x2]
                result.append({'roi': roi,
                               'lnRoi': ln_roi,
                               'x1': x1,
                               'x2': x2,
                               'y1': q['y1'],
                               'y2': q['y2'],
                               'flag': 'NL', })
                tmp = -9999
                meet_text = False
    return result


def non_line_cut_block(src_img, gray, out_folder):
    result = non_line_cut_hrz(src_img, gray)
    result = non_line_cut_vrt(result)
    i = 1
    for q in result:
        cv2.imwrite(
            out_folder + 'b' + str(i) + '-' + str(q['x1']) + '-' + str(q['x2']) + '-' + str(q['y1']) + '-' + str(
                q['y2']) + '.jpg', q['roi'])

        # res_obj = {
        #     "x1": q['x1'],
        #     "x2": q['x2'],
        #     "y1": q['y1'],
        #     "y2": q['y2'],
        #     "name": 'b' + str(i) + '-' + str(q['x1']) + '-' + str(q['x2']) + '-' + str(q['y1']) + '-' + str(
        #         q['y2']) + '.jpg'
        # }
        i += 1
        # print(json.dumps(res_obj))


def has_hrz_line(ln_roi):
    (h, w) = ln_roi.shape[:2]
    num_line = 0
    for k in range(h):
        row = ln_roi[k:k + 1, 0:w]
        if np.sum(row) > MIN_LEN_LN:
            num_line += 1
    if num_line > MIN_NUM_LN:
        return True
    return False


def has_vrt_line(ln_roi):
    (h, w) = ln_roi.shape[:2]
    num_line = 0
    for j in range(w):
        col = ln_roi[0:h, j:j + 1]
        if np.sum(col) > MIN_LEN_LN:
            num_line += 1
    if num_line > MIN_NUM_LN:
        return True
    return False


def doc_has_line(src_img_file):
    src = cv2.imread(src_img_file)
    (h, w) = src.shape[:2]
    if src is None:
        print("Problem loading image!!!\n")
        exit()

    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hrz_size = int(h / 30)
    vrt_size = int(w / 30)
    hrz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hrz_size, 1))
    vrt_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vrt_size))
    hrz_lines = cv2.erode(gray, hrz_kernel, iterations=1)
    hrz_lines = cv2.dilate(hrz_lines, hrz_kernel, iterations=1)
    vrt_lines = cv2.erode(gray, vrt_kernel, iterations=1)
    vrt_lines = cv2.dilate(vrt_lines, vrt_kernel, iterations=1)
    line_img = hrz_lines + vrt_lines
    # non_zero_pixels = cv2.countNonZero(line_img)
    # print (non_zero_pixels)
    if has_hrz_line(line_img) and has_vrt_line(line_img):
        return True, src, line_img, gray
    else:
        return False, src, line_img, gray


def cut_cell_all_cases(src_img_file, out_folder):
    has_line, src_img, line_img, gray = doc_has_line(src_img_file)
    cv2.imwrite(out_folder + 'gray.jpg', gray)
    if has_line:
        # print('has line')
        cv2.imwrite(out_folder + 'line.jpg', line_img)
        line_cut_queue(src_img, line_img, out_folder)
    else:
        # print('non line')
        cv2.imwrite(out_folder + 'non_line.jpg', line_img)
        non_line_cut_block(src_img, gray, out_folder)


# Below functions are currently unused

def cut_sub_block(ln_roi, roi, num_roi, out_folder):
    pre_x_pos = 0
    has_line = False
    (h, w) = roi.shape[:2]
    for x_pos in range(w):
        if np.sum(ln_roi[0:h, x_pos:x_pos + 1]) > 30000:
            if x_pos - pre_x_pos < GAP_TOL:
                continue
            sub_roi = roi[0:h, pre_x_pos:x_pos]
            cv2.imwrite(out_folder + str(num_roi) + 'sub_roi.jpg', sub_roi)
            pre_x_pos = x_pos
            has_line = True
            # print('has line %d' % num_roi)
            num_roi += 1
    if has_line is False:
        cv2.imwrite(out_folder + str(num_roi) + 'sub_roi.jpg', roi)
        num_roi += 1
    return num_roi


def cut_block(src_img_file, out_folder):
    src = cv2.imread(src_img_file)
    (h, w) = src.shape[:2]
    if src is None:
        print("Problem loading image!!!\n")
        exit()

    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hrz_size = int(h / 30)
    vrt_size = int(w / 30)
    hrz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hrz_size, 1))
    vrt_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vrt_size))
    hrz_lines = cv2.erode(gray, hrz_kernel, iterations=1)
    hrz_lines = cv2.dilate(hrz_lines, hrz_kernel, iterations=1)
    vrt_lines = cv2.erode(gray, vrt_kernel, iterations=1)
    vrt_lines = cv2.dilate(vrt_lines, vrt_kernel, iterations=1)
    line_img = hrz_lines + vrt_lines
    # cv2.imwrite(outPath, lineImg)
    cv2.imwrite('line_only.jpg', line_img)

    # Cutting block
    pre_y_pos = 0
    num_roi = 1
    for y_pos in range(h):
        if np.sum(line_img[y_pos:y_pos + 1, 0:w]) > 200000:
            if y_pos - pre_y_pos < GAP_TOL:
                continue
            roi = src[pre_y_pos:y_pos, 0:w]
            ln_roi = line_img[pre_y_pos:y_pos, 0:w]
            # cv2.imwrite("C:\\Users\\Thuong.Tran\\Desktop\\output\\" + str(t) + 'roi.jpg', lnRoi)
            num_roi = cut_sub_block(ln_roi, roi, num_roi, out_folder)
            pre_y_pos = y_pos


def detect_line(img_file, out_path):
    img = cv2.imread(img_file)
    (h, w) = img.shape[:2]
    if img is None:
        print("Problem loading image!!!\n")
        exit()

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hrz_size = int(h / 30)
    vrt_size = int(w / 30)
    hrz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hrz_size, 1))
    vrt_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vrt_size))
    hrz_lines = cv2.erode(gray, hrz_kernel, iterations=1)
    hrz_lines = cv2.dilate(hrz_lines, hrz_kernel, iterations=1)
    vrt_lines = cv2.erode(gray, vrt_kernel, iterations=1)
    vrt_lines = cv2.dilate(vrt_lines, vrt_kernel, iterations=1)
    cv2.imwrite(out_path, hrz_lines + vrt_lines)


def hist_pro(bn_img_file):
    img = cv2.imread(bn_img_file)
    sum_cols = vrt_pro(img)
    sum_rows = hrz_pro(img)
    return [sum_cols, sum_rows]


def vrt_pro(img):
    (h, w) = img.shape[:2]
    sum_cols = []
    for j in range(w):
        col = img[0:h, j:j + 1]  # y1:y2, x1:x2
        sum_cols.append(np.sum(col))
    return sum_cols


def hrz_pro(img):
    (h, w) = img.shape[:2]
    sum_rows = []
    for k in range(h):
        row = img[k:k + 1, 0:w]
        sum_rows.append(np.sum(row))
    return sum_rows