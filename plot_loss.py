#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
f = open('./result/loss_log.txt')
epoch_iter = 1
epochInd = -1
epoch = []
all_G_A, all_Cyc_A, all_D_A, all_idt_A = [], [], [], []
all_G_B, all_Cyc_B, all_D_B, all_idt_B = [], [], [], []
total_loss_list = []

WANG = True

if WANG:
    for line in f:
        if line[:6] == '(epoch':
            line_split = line.split(' ')
            if int(line_split[1].split(',')[0]) != epochInd:
                #print(line_split)
                epochInd = int(line_split[1].split(',')[0])
                # losstemp = (float(line_split[11].split(',')[0]) + float(line_split[13].split(',')[0])
                # + float(line_split[17].split(',')[0])
                #            +float(line_split[19].split(',')[0]) + float(line_split[21].split(',')[0])
                # + float(line_split[23].split(',')[0]))
                G_A = float(line_split[11].split(',')[0])
                D_A = float(line_split[9].split(',')[0])
                Cyc_A = float(line_split[13].split(',')[0])
                idt_A = float(line_split[15].split(',')[0])

                G_B = float(line_split[19].split(',')[0])
                D_B = float(line_split[17].split(',')[0])
                Cyc_B = float(line_split[21].split(',')[0])
                idt_B = float(line_split[23].split(',')[0])

                total_loss = G_A + D_A + Cyc_A + idt_A + G_B + D_B + Cyc_B + idt_B

                epoch.append(epoch_iter)
                epoch_iter += 1
                all_G_A.append(G_A)
                all_Cyc_A.append(Cyc_A)
                all_D_A.append(D_A)
                all_idt_A.append(idt_A)
                all_G_B.append(G_B)
                all_Cyc_B.append(Cyc_B)
                all_D_B.append(D_B)
                all_idt_B.append(idt_B)
                total_loss_list.append(total_loss)      # total loss of A-to-B
else:
    for line in f:
        if line[:6] == '(epoch':
            line_split = line.split(' ')
            if int(line_split[1].split(',')[0]) != epochInd:
                #print(line_split)
                epochInd = int(line_split[1].split(',')[0])
                #losstemp = (float(line_split[11].split(',')[0]) + float(line_split[13].split(',')[0])
                # + float(line_split[17].split(',')[0])
                #            +float(line_split[19].split(',')[0]) + float(line_split[21].split(',')[0])
                # + float(line_split[23].split(',')[0]))
                G_A = float(line_split[11].split(',')[0])
                D_A = float(line_split[9].split(',')[0])
                Cyc_A = float(line_split[13].split(',')[0])
                #idt_A = float(line_split[21].split(',')[0])
                idt_A = 0

                G_B = float(line_split[17].split(',')[0])
                D_B = float(line_split[15].split(',')[0])
                Cyc_B = float(line_split[19].split(',')[0])
                #idt_B = float(line_split[23].split(',')[0])
                idt_B = 0

                epoch.append(epoch_iter)
                epoch_iter += 1
                all_G_A.append(G_A)
                all_Cyc_A.append(Cyc_A)
                all_D_A.append(D_A)
                all_idt_A.append(idt_A)
                all_G_B.append(G_B)
                all_Cyc_B.append(Cyc_B)
                all_D_B.append(D_B)
                all_idt_B.append(idt_B)
f.close()

epoch = epoch[:200]
all_G_A = all_G_A[:200]
all_Cyc_A = all_Cyc_A[:200]
all_D_A = all_D_A[:200]
all_idt_A = all_idt_A[:200]
all_G_B = all_G_B[:200]
all_Cyc_B = all_Cyc_B[:200]
all_D_B = all_D_B[:200]
all_idt_B = all_idt_B[:200]


fig1 = plt.figure(figsize=(9, 6))
plt.subplot(2,2,1)
plt.plot(epoch, all_G_A)
plt.title('generator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,2)
plt.plot(epoch, all_D_A)
plt.title('discriminator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,3)
plt.plot(epoch, all_Cyc_A)
plt.title('cycle-consistency loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,4)
plt.plot(epoch, all_idt_A)
plt.title('identity loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.35)
#plt.tight_layout()
plt.suptitle('Class A (forward): Painting to Photo')
plt.savefig('./result/A_loss.jpg')

fig2 = plt.figure(figsize=(9, 6))
plt.subplot(2,2,1)
plt.plot(epoch, all_G_B)
plt.title('generator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,2)
plt.plot(epoch, all_D_B)
plt.title('discriminator loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,3)
plt.plot(epoch, all_Cyc_B)
plt.title('cycle-consistency loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(2,2,4)
plt.plot(epoch, all_idt_B)
plt.title('identity loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.35)
plt.suptitle('Class B (backward): Photo to Painting')
#plt.tight_layout()
plt.savefig('./result/B_loss.jpg')

fig3 = plt.figure()
plt.plot(epoch, total_loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Combined Loss of UNet (Forward)')
plt.savefig('./result/combined_loss')