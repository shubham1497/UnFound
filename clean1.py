from bs4 import BeautifulSoup
import re
import nltk 
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim

def textToWords(raw_review):

		review_text = BeautifulSoup(raw_review).get_text()

		letters_only = re.sub('[^a-zA-Z]+', ' ', review_text)

		words = letters_only.lower().split()

		stop_words = set(stopwords.words('english'))
		meaningful_words = [w for w in words if not w in stop_words]

		long_words = [w for w in meaningful_words if len(w) >= 3]

		return(' '.join(long_words))


def getAllTextInput(dir_name):
		clean_train_reviews = []
		i=1
		for filename in os.listdir(dir_name):
				current=os.path.join(dir_name, filename)
				if(os.path.isfile(current)):
						inFile=open(current,encoding="utf8",mode ='r')
						raw_review = inFile.read()
						clean_train_reviews.append(textToWords(raw_review))
						print ('File'+str(i)+' proccessed\n')
						i=i+1
		return clean_train_reviews

if __name__=='__main__':
		path = input('Please enter location of dir where data is stored :')
		
		clean_review_text = getAllTextInput(path)
		tokens = []
		for f in clean_review_text:
			token = []
			for i in word_tokenize(f):
				token.append(i)
			tokens.append(token)
		dictionary = corpora.Dictionary(tokens)
    
		corpus = [dictionary.doc2bow(text) for text in tokens]

		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=100)
		latent = ldamodel.print_topics(num_topics=5, num_words=5)
		print(latent)
		
		