import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

ulasans = pd.read_csv('../data/dataset.csv')
fiturs = ulasans.iloc[:, 0].values
labels = ulasans.iloc[:, 1].values
class_counts = ulasans.groupby('Label').size()
print(class_counts)
print(ulasans.head())

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory1 = StemmerFactory()
stemmer = factory1.create_stemmer()

kalimats = []
for kalima in tqdm(range(0, len(fiturs))):
	kalimat = re.sub(r'\W', ' ', str(fiturs[kalima]))
	kalimat = re.sub(r'\s+[a-zA-Z]\s+', ' ', kalimat)
	kalimat = re.sub(r"\d+", "", kalimat.lower())
	kalimat = kalimat.translate(str.maketrans("","",string.punctuation))
	kalimat = ' '.join(kalimat.split())
	kalimat = stopword.remove(kalimat)
	kalimat = stemmer.stem(kalimat)
	kalimats.append(kalimat)
	
#def tocsv(arrnilais,labels):
#	df = pd.DataFrame({'ulasan':arrnilais,'label':labels})
#	df.to_csv ('dataset.csv', index = False, header=True)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(kalimats)
kamus = dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))
c = 0
for i in kalimats:
	a = str(i).split(' ')
	c = c + len(a)
#e = int(c/len(kalimats))
e = 50 # e is a row length
arrnilais=[]
for i in kalimats:
	arrnilai = []
	a = str(i).split(' ')
	b = e - len(a)
	if len(a) < e:        a.extend([0]*b)
	elif len(a) > e:        del a[b:]
	for j in a:
	    if j in kamus :            nilai = kamus.get(j)
	    else :            nilai = 0
	    arrnilai.append(float(nilai))
	arrnilais.append(arrnilai)
arrnilais = np.asarray(arrnilais)