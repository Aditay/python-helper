all_text=open('utt2trans.txt', 'r')

a= []

for trans in all_text:
	utt = trans.split(' ')
	a.append(utt)

