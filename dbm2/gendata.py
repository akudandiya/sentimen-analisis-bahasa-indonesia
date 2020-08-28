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
	
kalimats = np.asarray(kalimats)
arrnil=[]
c = 0
for i in tqdm(kalimats) :
	kalimat = re.sub(r'\s', '', i)
	kalimat = ''.join(format(ord(x), 'b') for x in kalimat)
	kalimat = ''.join(kalimat)
	c = c + len(kalimat)
	arrnil.append(kalimat)
arrnil = np.array(arrnil)
#e = int(c/len(kalimats))
e = int(50)
arrnilais=[]
for j in tqdm(arrnil):
    a = [int(i) for i in j]
    b = e - len(a)
    if len(a) < e:        a.extend([0]*b)
    elif len(a) > e:        del a[b:]
    arrnilais.append(a)
arrnilais=np.asarray(arrnilais)