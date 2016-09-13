def keep_alpha_digit(word):
	new_word=''
	for i in range(len(word)):
		if str(word[i]).isalpha() or str(word[i]).isdigit():
			new_word=new_word+str(word[i])

	return new_word


def stemming(word):
	return word


def proc_word(word):

	word=keep_alpha_digit(word)
	word=stemming(word)

	return word
