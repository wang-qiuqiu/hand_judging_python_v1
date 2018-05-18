# -*- coding: utf-8 -*
import time
import math
import cv2
import scipy.io as sio
from pylab import *
from statistics import *
import numpy as np
from skimage import measure
from matplotlib.font_manager import FontProperties
from skimage import morphology

def find(matrix,x):
	row = []
	col = []
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] == x:
				row.append(i)
				col.append(j)
	return row,col

def autoroi(bwimage):
    row,col = find(bwimage,1)
    y_roi = min(col,default=0)
    x_roi = min(row,default=0)
    try:
        width_roi = max(col)-min(col,default=0)
    except ValueError:
        width_roi = 0
    try:
        height_roi = max(row)-min(row,default=0)
    except ValueError:
        height_roi = 0
    if len(row) == 0:
        hand_roi = bwimage
        w = len(bwimage[0])
        h = len(bwimage)
    else:
        hand_roi = bwimage[x_roi:x_roi+height_roi,y_roi:y_roi+width_roi]
        w = width_roi
        h = height_roi
    return hand_roi,w,h

def  Comp_effluent(IM,I_t,thresh,x,y):
    IM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
    I_t = cv2.cvtColor(I_t,cv2.COLOR_BGR2GRAY)
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]- x[0]) + 1
    I_tc = I_t[x_point:x_point + height,y_point:y_point + width]
    box = IM[x_point:x_point + height,y_point:y_point + width]
    diff = double((abs(double(box)-double(I_tc)))).astype(uint8)
    retval,bw = cv2.threshold(diff,thresh*255,1, cv2.THRESH_BINARY)
    bw = np.array(bw)
    score = np.sum(bw==1)/bw.size
    return score

def Judge_effluent(score,mint,maxt):
	if score>mint and score<maxt:
		result = 1
	elif score>=maxt:
		result = 1
	else:
		result = 0
	return result

def Comp_wash(I_t,IM,bwt,dlt,x,y):
    IM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
    I_t = cv2.cvtColor(I_t,cv2.COLOR_BGR2GRAY)
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    diff = (abs(double(IM)-double(I_t))).astype(uint8)
    retval,bw = cv2.threshold(diff,bwt*255,1,cv2.THRESH_BINARY)
    bw = morphology.remove_small_objects(bw,min_size=dlt,connectivity=2)
    #对ROI区域进行裁剪
    box = bw[x_point:x_point+height,y_point:y_point+width]
    return box

def Judge_wash(c,j,ef):
    if j <= ef:
        try:
            if mode(c[:j]) == 1:
                result = 1
            else:
                result = 0
        except StatisticsError:
            result =  0
    else:
        if mode(c[j-ef-1:j]) == 1:
            result = 1
        else:
            result = 0
    return result

def Comp_soap(IM,I_t,lt,x,y):
    I_t = cv2.cvtColor(I_t,cv2.COLOR_BGR2GRAY)
    IM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    I_tc = I_t[x_point:x_point+height,y_point:y_point+width]
    box = IM[x_point:x_point+height,y_point:y_point+width]
    diff = cv2.absdiff(box,I_tc)
    diff = np.array(diff)
    rows = len(diff)
    cols = len(diff[0])
    size = I_tc.size
    hl = 0
    for i in range(rows):
        for j in range(cols):
            if diff[i][j] > lt:
                hl += 1
    h1 = hl/size
    return h1

def Judge_soap(c_sp,j,fps,tt):
	if c_sp[j] == 1:
		result = 1
	else:
		if (len([i for i in c_sp[:j] if i == 1])/fps) < tt:
			result = 0
		else:
			result = 3
	return result

def Comp_foam(IM,I_t,bwt,dlt,x,y):
    I_t = cv2.cvtColor(I_t,cv2.COLOR_BGR2GRAY)
    IM = cv2.cvtColor(IM,cv2.COLOR_BGR2GRAY)
    x_point = int(x[0])
    y_point = int(y[0])
    width =  int(y[1]-y[0]) + 1
    height = int(x[1]-x[0]) + 1
    I_tc = I_t[x_point:x_point+height,y_point:y_point+width]
    box = IM[x_point:x_point+height,y_point:y_point+width]
    diff = (abs(double(box)-double(I_tc))).astype(uint8)
    retval,bw = cv2.threshold(diff,bwt*255,1, cv2.THRESH_BINARY)
    bw = morphology.remove_small_objects(bw,min_size=dlt,connectivity=2)
    hand_roi = autoroi(bw)
    labels = measure.label(hand_roi[0],connectivity=1)
    num = labels.max()+1
    return num,hand_roi

def ShowResult(result):
    if result == 1:
        word = 'yes'
    elif result == 2:
        word = 'cannot judge'
    elif result == 3:
        word = 'no(finished)'
    else:
        word = 'no'
    return word

def Judge_washcurrent(hand):
    col = []
    for i in range(len(hand)):
        for j in range(len(hand[0])):
            if hand[i][j] == 1:
                col.append(j)
    if len(col) == 0:
        c = 0
    elif min(col)<(0.5*len(hand[0])+5):
        c = 1
    else:
        c = 0
    return c


if __name__ == "__main__":
    time_start = time.time()

    video_full_path = "Video\\total.mp4"
    capture = cv2.VideoCapture(video_full_path)

    # opencv2.4.9用cv2.cv.CV_CAP_PROP_FPS；如果是3用cv2.CAP_PROP_FPS
    (major_ver,minor_ver,subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)

    #获取所有帧
    frame_count = 0
    all_frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1

    #I_t为第一帧也就是背景
    I_t = all_frames[0]

    #设置参数
    # time = 0
    stf = 1
    ovf = frame_count
    min_et = 0.2
    max_et = 0.4
    bwt_et = 50/255
    bwt_ws = 50/255
    dlt_ws = 10
    ef_ws = 20
    lr_sp = 0.02
    tt_sp = 1
    lt_sp = 100
    ef_fm = math.ceil(fps)*2
    bwt_fm = 50/255
    dlt_fm = 5
    crnt_fm = 5
    pfnt_fm = 40
    tpft_fm = 3

    #获取ROI区域中的八个坐标点
    data = sio.loadmat('ROI_coordinate.mat')
    x_et = data['x_et']
    y_et = data['y_et']
    x_ws = data['x_ws']
    y_ws = data['y_ws']
    x_sp = data['x_sp']
    y_sp = data['y_sp']
    x_fm = data['x_fm']
    y_fm = data['y_fm']
	
	#初始化数组长度为所有帧的个数
    s_et = [0 for i in range(len(all_frames))]
    result_ws = [0 for i in range(len(all_frames))]
    result_et = [0 for i in range(len(all_frames))]
    result_fm = [0 for i in range(len(all_frames))]
    c_w = [0 for i in range(len(all_frames))]
    c_sp = [0 for i in range(len(all_frames))]
    result_sp = [0 for i in range(len(all_frames))]
    crn_fm = [0 for i in range(len(all_frames))]
    pfn_fm = [0 for i in range(len(all_frames))]

    hl = [0 for i in range(len(all_frames))]

    cv2.namedWindow("Result")
    for i in range(len(all_frames)):
        IM = all_frames[i]
        j = i-(stf-1)

        #水流判断
        s_et[j] = Comp_effluent(IM,I_t,bwt_et,x_et,y_et)
        result_et[j] = Judge_effluent(s_et[j],min_et,max_et)

        #手在水中判断
        hand = Comp_wash(I_t,IM,bwt_ws,dlt_ws,x_ws,y_ws)
        c_w[j] = Judge_washcurrent(hand)
        result_ws[j] = Judge_wash(c_w,j,ef_ws)

        #洗手液判断
        # time_ws = result_ws.count(1)/fps
        hl[j] = Comp_soap(IM,I_t,lt_sp,x_sp,y_sp)
        hl = np.array(hl)
        c_sp = np.array(c_sp)
        if hl[j] > lr_sp:
            c_sp[j] = 1
        else:
            c_sp[j] = 0
        result_sp[j] = Judge_soap(c_sp, j, fps, tt_sp)

        #泡沫判断
        crn_fm[j],hand_roi = Comp_foam(IM,I_t,bwt_fm,dlt_fm,x_fm,y_fm)
        if j<=ef_fm:
            pfn_fm[j] = len([i for i in crn_fm[:j] if i>crnt_fm])
        else:
            pfn_fm[j] = len([i for i in crn_fm[j-ef_fm-1:j] if i>crnt_fm])
        if len([i for i in pfn_fm if i>pfnt_fm]) > tpft_fm:
            result_fm[j] = 1
        else:
            result_fm[j] = 0

        word_et = ShowResult(result_et[j])
        word_ws = ShowResult(result_ws[j])
        word_sp = ShowResult(result_sp[j])
        word_fm = ShowResult(result_fm[j])

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(IM, 'pic:%s'%(str(j))+' '+'Water:%s'%(str(word_et))+
                    ' '+'Hand:%s'%str(word_ws) +
                    ' ' + 'Soap:%s'%str(word_sp) +
                    ' '+ 'Foam:%s'%str(word_fm),
                    (30, 500), font, 1, (255, 0, 255), 2)

        cv2.imshow("Result", IM)
        c = cv2.waitKey(5)

    time_end = time.time()
    time_cost = time_end - time_start
    print(time_cost)