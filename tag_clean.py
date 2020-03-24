for group in ['train']:
	for level in ['dif','simp']:
		fname = f'{group}.{level}'
		raw_t = open(fname).read().split('\n')
		tag_t = open('ts_tag/'+fname).read().split('\n')
		count = 0
		new_tag = []
		for line in raw_t:
			if line[-1] == '.' or line[-1] == '?' or line[-1] == '!':
				if tag_t[count][-1] != line[-1]:
					new_tag[-1] += ' ' + tag_t[count]
					count += 1
					continue
			new_tag.append(tag_t[count])
			count += 1
		open('ts_tag3/'+fname,'w').write('\n'.join(new_tag))
					

			
		