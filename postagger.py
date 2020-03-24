import stanfordnlp

nlp = stanfordnlp.Pipeline(processors = 'tokenize,mwt,pos')

def parse(sents):
	re_list = []
	sents = sents.split('\n')

	count = 0
	for sent in sents:
		tags = []
		texts = []

		try:
			
			for s in nlp(sent).sentences:
				tags.extend([[word.text,word.xpos] for word in s.words])
		except Exception as e:
			print(count,e)
			tags = []
		count += 1
		re_list.append(tags)
	return re_list
