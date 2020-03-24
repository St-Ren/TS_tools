from random import randint

lines = open('../newsela.sents.txt',encoding = 'utf-8')
trainx = []
trainy = []
testx = []
testy = []
validx = []
validy = []
docnum=0
for line in lines:
	
	line = line.split('\t')
	x = line[3]
	y = line[4]
	tmp_doc = int(line[0].split('OC')[1])
	src = int(line[1][1])
	tgt = int(line[2][1])
	if tgt-src == 1:
		continue
	if 1:#x not in trainx and x not in testx and x not in validx:
		if docnum != tmp_doc:
				rdnum = randint(1,10)
		if rdnum == 10:
			testx.append(x)
			testy.append(y)
		elif rdnum == 9:
			validx.append(x)
			validy.append(y)
		else:
			trainx.append(x)
			trainy.append(y)
open('test.dif','w',encoding='utf-8').write('\n.'.join(testx))
open('test.simp','w',encoding='utf-8').write(''.join(testy))
open('train.dif','w',encoding='utf-8').write('\n'.join(trainx))
open('train.simp','w',encoding='utf-8').write(''.join(trainy))
open('valid.dif','w',encoding='utf-8').write('\n'.join(validx))
open('valid.simp','w',encoding='utf-8').write(''.join(validy))