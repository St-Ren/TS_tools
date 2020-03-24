import json 


import postagger

for group in ['test','valid']:
	for level in ['simp','dif']:
		print("group:{}, level:{}".format(group,level))
		text = open('{}.{}'.format(group,level)).read()
		fname = f'{group}.{level}'
		text_tags = postagger.parse(text)
		text = '\n'.join([' '.join([tag[1] for tag in line_tags]) for line_tags in text_tags])
		open('ts_tag/'+fname,'w').write(text)
		if level == 'simp':
			raw = '\n'.join([' '.join([tag[0] for tag in line_tags]) for line_tags in text_tags])
		else:
			raw = '\n'.join([' '.join([tag[1]+'/'+tag[0] for tag in line_tags]) for line_tags in text_tags])

		open('ts_tag/raw/'+fname,'w').write(raw)
