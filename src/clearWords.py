from unicodedata import normalize
import json
import nltk
import re
import emoji
import itertools
import os
from pathlib import Path

class ClearWords():
    def __init__(self, df):
        self.df = df

    def lowerCaseWords(self, label):
       self.df[label] = self.df[label].apply(lambda x: x.lower() if isinstance(x, str) else x)
       return self.df[label]

    def removeStopWords(self, label):
        try:
            with open(os.path.join(Path().absolute().parent) + '\\stopwords', 'r+', encoding='utf-8') as f:
                stopwords = [line[:-(line[-1] == '\n') or len(line)+1].strip() for line in f]
        except FileNotFoundError:
            stopwords = [normalize('NFKD', w).encode('ASCII', 'ignore').decode('ASCII') for w in nltk.corpus.stopwords.words('portuguese')]

        stopwords.extend(['pra', 'pro', 'muito', 'muita', 'pq', 'ai', 'coisa', 'tipo', 'desse', 'dessa', 
        'nesse', 'nessa', 'onde', 'assim', 'porque', 'por', 'que', 'deixar', 'voltar', 
        'fazendo', 'quanto', 'parece', 'toda', 'ficou', 'deu', 'fica','precisa','fazer', 'estar', 'sendo', 
        'espero', 'posso', 'disse', 'fico', 'hoje', 'ontem', 'amanha', 'manha', 'tarde', 'noite', 'dia', 
        'querer', 'aqui', 'ainda', 'todos', 'pois', 'outra', 'parte', 'sai', 'voce',
        'falar', 'nova', 'novo', 'junto', 'ter', 'vc', 'viu', 'vio', 'ate', 'cade', 'jeito', 'sido'])

        self.df[label] = self.df[label].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]).strip())
        return self.df[label]

    def removeSlang(self, label):
        with open(os.path.join(Path().absolute().parent) + '\\termos.json', encoding='utf-8') as json_file:
            slang = json.load(json_file)

        self.df[label] = self.df[label].apply(lambda x: ' '.join(slang.get(ele, ele) for ele in x.split()))
        return self.df[label]

    def removerAcentos(self, label):
        self.df[label] = self.df[label].apply(lambda x: ' '.join([normalize('NFKD', w)\
                                                                  .encode('ASCII', 'ignore')\
                                                                  .decode('ASCII') for w in x.split()])\
                                                                  .strip())
        return self.df[label]
    
    def emoji2text(self, label):
        self.df[label] = self.df[label].apply(lambda x: ' '.join([re.sub(":", " ", w) if re.search(r"^(\S+)\s*\S*(?=:).*:$", w) else w for w in emoji.demojize(x, language="pt").split()]))
        return self.df[label]

    def removeSingleWord(self, label, size=2):
        self.df[label] = self.df[label].apply(lambda x: ' '.join([w for w in x.split() if len(w) > size]).strip())
        return self.df[label]

    def createRads(self, label):
        self.df[label] = self.df[label].apply(lambda x: ' '.join([str(nltk.stem.RSLPStemmer().stem(w)) for w in x.split()]))
        return self.df[label]

    def removeCharacters(self, label, n=2, url=None):
        lst = []
        for i in self.df[label]:
            try:
                if url: r = '(http|www|bit)(\S+|.)|\W+|_+|[0-9]+'
                else: r = '\W+|_+|[0-9]+'
                rex = [re.sub(r, '', k).lower() for k in i.split(' ')]
                lst.append(' '.join(filter(lambda x: len(x) > n, rex)).strip())
            except AttributeError:
                print(i)
        self.df[label] = lst
        return self.df[label]

    def removerLetrasDuplicadas(self, label, size=3):
        r = "(.)\\1{" + str(size - 1) + ",}"
        lst = []
        for i in self.df[label]: 
            for x in re.findall(r, i):
                i = re.sub(f"({x})" + "\\1{" + str(size - 1) + ",}", x, i)
            lst.append(i)
        self.df[label] = lst
        return self.df[label]

    def removeEquals(self, label):
        lst = []
        for i in self.df[label]:
            try:
                lst.append(' '.join(list(zip(*itertools.groupby(i.split())))[0]))
            except IndexError:
                lst.append(i)
        self.df[label] = lst
        return self.df[label]

    def removeMentions(self, label):
         self.df[label] = self.df[label].apply(lambda x: ' '.join([re.sub('^@[A-Za-z0-9].*', '', k).lower() for k in x.split(' ')]).strip())
         return self.df[label]
