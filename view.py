import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cbook as cbook
import csv
import cv2
import os
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import rotem_helpers

dataset_path = '/Users/rotemisraeli/Documents/datasets/tennis/'
game = 8
clip = 1
main_dir = dataset_path+'game'+str(game)+'/' + str(clip).zfill(2) + '/'
image = 17
video_name = 'video3'

def main():
	# view_single_image(image,label_dir=main_dir + 'Label.csv')
	# save_video(label_dir=main_dir + 'Label.csv')
	view_image_from_model()


def view_single_image(image,label_dir,save=-1):
	label_file = open(label_dir, 'r')
	label_list = list(csv.reader(label_file))
	label_line = label_list[image+1]

	image_file = cbook.get_sample_data(main_dir + str(image).zfill(4) + '.jpg')
	img = plt.imread(image_file)
	fig, ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(img)

	try:
		ball_x = int(label_line[2])
		ball_y = int(label_line[3])
		ball_status = int(label_line[4])
		if ball_status == 0:
			circ = Circle((ball_x, ball_y), 5, color=[0.2, 1, 0.3, 1])
		elif ball_status == 1:
			circ = Circle((ball_x, ball_y), 5, color=[1, 0.1, 0.2, 1])
		else:
			circ = Circle((ball_x, ball_y), 5, color=[0.2, 0.6, 1, 1])
		ax.add_patch(circ)
	except:
		pass

	if save!=-1:
		plt.savefig(save)
	else:
		plt.show()

def view_image_from_model(save = -1):
	image_test = 0
	net = torch.load('regression2.torch')
	x_test = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/x_test.np')
	y_test = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/y_test.np')
	x_test = torch.from_numpy(np.asarray(x_test, dtype=float)).float()
	y_test = torch.from_numpy(np.asarray(y_test, dtype=float)).float()

	output = net(x_test)
	print(y_test[image_test])
	print(output[0])

	image_file = cbook.get_sample_data(main_dir + str(image).zfill(4) + '.jpg')
	img = plt.imread(image_file)
	fig, ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(img)

	try:
		for i in range(20):
			ball_x = int(y_test[image_test][i*2])
			ball_y = int(y_test[image_test][i*2+1])
			circ = Circle((ball_x, ball_y), 5, color=[0.2, 1, 0.3, 1])
			ax.add_patch(circ)

			if i not in (18,17,16,13,11):
				ball_x = int(output[image_test][i*2])
				ball_y = int(output[image_test][i*2+1])
				circ = Circle((ball_x, ball_y), 5, color=[0.2, 0.6, 1, 1])
				ax.add_patch(circ)
	except:
		pass

	if save != -1:
		plt.savefig(save)
	else:
		plt.show()


def save_video(label_dir=-1):
	video_path = '/Users/rotemisraeli/Documents/python/tennis/'+video_name+'.avi'
	if label_dir == -1:
		images_to_video(main_dir,video_path)
	else:
		dir_path = './game' + str(game) + '-' + str(clip).zfill(2)
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
		images = sorted([img for img in os.listdir(main_dir) if img.endswith(".jpg")])
		for i in tqdm(range(len(images))):
			view_single_image(i,label_dir,save=dir_path+'/'+str(i).zfill(4)+'.jpg')
		images_to_video(dir_path,video_path)


def images_to_video(images_dir,video_path):
	images = sorted([img for img in os.listdir(images_dir) if img.endswith(".jpg")])
	frame = cv2.imread(os.path.join(images_dir, images[0]))
	height, width, layers = frame.shape

	fourcc = cv2.VideoWriter.fourcc('H', '2', '6', '4');

	video = cv2.VideoWriter(video_path, fourcc, 12, (width, height))

	for image in images:
		video.write(cv2.imread(os.path.join(images_dir, image)))

	cv2.destroyAllWindows()
	video.release()

if __name__ == '__main__':
    main()