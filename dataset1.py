import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cbook as cbook
import csv
import cv2
import os
from tqdm import tqdm
import rotem_helpers
# x: few points from the hit
# y: the point it will touch the ground


dataset_path = '/Users/rotemisraeli/Documents/datasets/tennis/'


def main():
	build_dataset2()
	print(len(x_train))
	print(len(x_test))
	print(len(y_train))
	print(len(y_test))
	print('\n')
	for i in range(len(y_train[2])):
		if i % 2 == 0:
			print(y_train[0][i],y_train[0][1])

shots_input = 4
x_train,y_train,x_test,y_test = [],[],[],[]


def single_clip1(label_list,test):
	global x_train,y_train,x_test,y_test
	bounces = []
	hits = []
	for i,row in enumerate(label_list):
		if row[4] == '2':
			bounces.append(i)
		if row[4] == '1':
			hits.append(i)

	for b in bounces:
		if hits and b > hits[0]:
			hit = -1
			for h in hits:
				if h < b:
					hit = h
				else:
					break
			if b - hit < 45 and b - hit > shots_input:
				try:
					input1 = []
					for i in range(hit,hit+4):
						input1.append(int(label_list[i][2]))
						input1.append(int(label_list[i][3]))
					if test:
						x_test.append(input1)
						y_test.append([int(label_list[b][2]),int(label_list[b][3])])
					else:
						x_train.append(input1)
						y_train.append([int(label_list[b][2]),int(label_list[b][3])])
				except:
					pass
def single_clip2(label_list,test):
	global x_train,y_train,x_test,y_test
	bounces = []
	hits = []
	for i,row in enumerate(label_list):
		if row[4] == '2':
			bounces.append(i)
		if row[4] == '1':
			hits.append(i)

	for b in bounces:
		if hits and b > hits[0]:
			hit = -1
			for h in hits:
				if h < b:
					hit = h
				else:
					break
			if b - hit < 45 and b - hit > shots_input:
				try:
					input1 = []
					output1 = []
					for i in range(hit,hit+4):
						input1.append(int(label_list[i][2]))
						input1.append(int(label_list[i][3]))
					for i in range(hit, hit + 45):
						if i < b:
							output1.append(int(label_list[i][2]))
							output1.append(int(label_list[i][3]))
						else:
							output1.append(int(label_list[b][2]))
							output1.append(int(label_list[b][3]))
					if test:
						x_test.append(input1)
						y_test.append(output1)
					else:
						x_train.append(input1)
						y_train.append(output1)
				except:
					pass


def build_dataset1():
	games = sorted([o for o in os.listdir(dataset_path)
	 if os.path.isdir(os.path.join(dataset_path, o))])
	games.append(games.pop(1))
	for game in games:
		clips = sorted([o for o in os.listdir(dataset_path+game)
						if os.path.isdir(os.path.join(dataset_path+game, o))])
		for clip in clips:
			label_file = open(dataset_path+game+'/'+clip+'/'+'Label.csv', 'r')
			label_list = list(csv.reader(label_file))
			single_clip1(label_list, test=game=='game9'or game=='game8')
	rotem_helpers.save_obj(x_train, '/Users/rotemisraeli/Documents/python/tennis/dataset1/x_train.np')
	rotem_helpers.save_obj(y_train, '/Users/rotemisraeli/Documents/python/tennis/dataset1/y_train.np')
	rotem_helpers.save_obj(x_test, '/Users/rotemisraeli/Documents/python/tennis/dataset1/x_test.np')
	rotem_helpers.save_obj(y_test, '/Users/rotemisraeli/Documents/python/tennis/dataset1/y_test.np')

def build_dataset2():
	games = sorted([o for o in os.listdir(dataset_path)
	 if os.path.isdir(os.path.join(dataset_path, o))])
	games.append(games.pop(1))
	for game in games:
		clips = sorted([o for o in os.listdir(dataset_path+game)
						if os.path.isdir(os.path.join(dataset_path+game, o))])
		for clip in clips:
			label_file = open(dataset_path+game+'/'+clip+'/'+'Label.csv', 'r')
			label_list = list(csv.reader(label_file))
			single_clip2(label_list, test=game=='game9'or game=='game8')
	rotem_helpers.save_obj(x_train, '/Users/rotemisraeli/Documents/python/tennis/dataset2/x_train.np')
	rotem_helpers.save_obj(y_train, '/Users/rotemisraeli/Documents/python/tennis/dataset2/y_train.np')
	rotem_helpers.save_obj(x_test, '/Users/rotemisraeli/Documents/python/tennis/dataset2/x_test.np')
	rotem_helpers.save_obj(y_test, '/Users/rotemisraeli/Documents/python/tennis/dataset2/y_test.np')

if __name__ == '__main__':
    main()
