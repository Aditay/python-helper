ali_dir=open('/media/Drive/kaldi/egs/wsj/s5/exp/my_tri1_libri_ali/merged_labels.txt', 'r')

root='/media/Drive/libri_labels_using_wsj/'

for alignment in ali_dir:
	labels=alignment.split(' ')
	file_name=labels[0]
	target_file=open(root+file_name+'.txt', 'w')
	for label in labels[1:-2]:
		target_file.write(label)
		target_file.write('\n')
	# target_file.write
	target_file.write(labels[-2])
	target_file.close()
