#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:04:24 2018

@author: zhonghao
"""

import matplotlib.pyplot as plt
f = open('loss_log.txt')
epoch_iter = 1
epochInd = -1
epoch = []
loss = []

for line in f:
    if line[:6] == '(epoch':
        line_split = line.split(' ')
        if int(line_split[1].split(',')[0]) != epochInd:
            #print(line_split)
            epochInd = int(line_split[1].split(',')[0])
            #losstemp = (float(line_split[11].split(',')[0]) + float(line_split[13].split(',')[0]) + float(line_split[17].split(',')[0])
            #            +float(line_split[19].split(',')[0]) + float(line_split[21].split(',')[0]) + float(line_split[23].split(',')[0]))
            losstemp = float(line_split[21].split(',')[0])
            epoch.append(epoch_iter)
            epoch_iter += 1
            loss.append(losstemp)
f.close()
#print(epoch, loss)
plt.figure(figsize=(12, 8))
plt.plot(epoch, loss)
plt.xlabel('epochs')
plt.ylabel('Forward cycle loss')
plt.savefig('A_identity_loss_vs_epoch.jpg')