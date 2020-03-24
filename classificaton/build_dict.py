import json 
tag_dict = dict()
mxlen = 0
for group in ['train','test','valid']:
	for level in range(5):
		fname = f'{group}{level}'
		text_tags = json.load(open(fname+'.tag'))
		for line_tags in text_tags:
			if len(line_tags)>mxlen:
				mxlen = len(line_tags)
			for tag in line_tags:
				if tag not in tag_dict:
					tag_dict[tag] = 0
				tag_dict[tag] +=1
dic = dict()
count = 0
for tag in tag_dict:
	dic[tag] = count
	count += 1
print(count)
print(mxlen)
json.dump(dic,open('tag_dic.txt','w'))