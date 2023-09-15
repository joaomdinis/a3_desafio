import pandas as pd
import numpy as np
import yaml, os, re, json
from pathlib import Path
from src.clearWords import ClearWords
from src.mariTalkFewShot import MariTalkFewShot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from itertools import chain, product
from time import sleep


class Main():
    def __init__(self, cfg ='config.yaml', df = 'churn_com_texto.csv', index_train=None, index_classifier=None):
        self.cfg = cfg
        self.df = df
        self.index_train = index_train
        self.index_classifier = index_classifier

    def _clearWords(self) -> pd.DataFrame:
        metodos = ['lowerCaseWords', 'removeStopWords', 'removeSlang', 'removerAcentos', 'removeStopWords', 'removeEquals', 'removerLetrasDuplicadas', 'removeCharacters']
        
        temp = self._readDataset()
        temp['Comentários'] = temp['Comentários'].fillna(temp['Número de Reclamações'])
        temp['Número de Reclamações'] = temp['Número de Reclamações'].apply(lambda x: re.sub('^.*(?!(^(\d|-)$)).$|2', '1', x)).str.replace('-', '0')
        temp['Volume de Dados'] = temp['Volume de Dados'].fillna('-').apply(lambda x: re.sub('(?i)^(?!(^\d.+gb$)).*$', '-', x))
        
        temp.loc[:,'Comentários Tratado'] = temp['Comentários']
        
        cl = ClearWords(temp)
        temp.loc[:,'Comentários Tratado'] = [getattr(cl, mtd)('Comentários Tratado') for mtd in metodos][-1]
        return temp

    def _readAPIKey(self) -> dict:
        with open(os.path.join(Path().absolute()) + f"\\conf\\{self.cfg}", "r") as ymlfile:
            return yaml.load(ymlfile, Loader=yaml.FullLoader)['mariTalk']['api_key']

    def _readDataset(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(Path().absolute()) + f"\\data\\{self.df}", dtype=str, on_bad_lines='skip')

    def _trainTestSplit(self) -> list:
        temp = self._clearWords()
        return train_test_split(temp, test_size=30, stratify=temp['Número de Reclamações'])

    def _creatDicts(self, parametersSearch = True) -> list:
        if parametersSearch: train, test = self._trainTestSplit()
        else: 
            temp = self._clearWords()
            train = temp.loc[self.index_train].sample(30)
            test = temp.loc[self.index_classifier]
        test = [{"role": "user", "content": "Dado o seguinte texto, classifique em uma das categorias: [positivo (0), negativo (1)]. Use metodologias de processamento de linguagem natural para determinar a categoria mais adequada e retorne apenas o valor classificado: " + str(i[1]), "Classificação Original": i[0], 'ID': i[2]} for i in test[['Número de Reclamações', 'Comentários Tratado', 'ID']].values]
        train = list(chain(*[[{"role": "user", "content": "Dado o seguinte texto, classifique em uma das categorias: [positivo (0), negativo (1)]. Use metodologias de processamento de linguagem natural para determinar a categoria mais adequada: " + str(i[1])}, {"role": "assistant", "content": str(i[0])}] for i in train[['Número de Reclamações', 'Comentários Tratado']].values]))
        return [train, test]

    def searchParametersMariTalk(self, **kwargs) -> dict:
        train, test = self._creatDicts()
        d = {'comentario_tratado': [], 'classificacao_original': [], 'id': [], 'classificacao_modelo': []}
        for i in np.array_split(train, 10):
            i = list(i)
            for j in test:
                try:
                    d['comentario_tratado'].append(j['content'].split(':')[-1])
                    d['classificacao_original'].append(j.pop('Classificação Original'))
                    d['id'].append(j.pop('ID'))
                    i.append(j)
                    d['classificacao_modelo'].append(MariTalkFewShot(self._readAPIKey(), 
                                                                    i, 
                                                                    max_tokens=kwargs.get('max_tokens'), 
                                                                    do_sample=kwargs.get('do_sample'), 
                                                                    temperature=kwargs.get('temperature'), 
                                                                    top_p=kwargs.get('top_p')
                                                                    ).predict()['answer'])
                except KeyError:
                    pass
                sleep(5) #limite de uma solicitação a cada 5 segundos
        return d

    def mariTalkClassifier(self, **kwargs) -> dict:
        train, test = self._creatDicts(parametersSearch=False)
        d = {'comentario_tratado': [], 'id': [], 'classificacao_modelo': []}

        for message in test:          
            temp = train
            try:
                d['comentario_tratado'].append(message['content'].split(':')[-1])
                d['id'].append(message.pop('ID'))
                temp.append(message)
                d['classificacao_modelo'].append(MariTalkFewShot(self._readAPIKey(), 
                                                                    temp, 
                                                                    max_tokens=kwargs.get('max_tokens'), 
                                                                    do_sample=kwargs.get('do_sample'), 
                                                                    temperature=kwargs.get('temperature'), 
                                                                    top_p=kwargs.get('top_p')
                                                                ).predict()['answer'])
            except KeyError as e:
                print(e)
                print(test)
            sleep(5) #limite de uma solicitação a cada 5 segundos
        return d

if __name__ == '__main__':
    ## Ajuste de parâmetros para few-shot classification com MariTalk
    parm_var = []
    parm_grid = {'max_tokens': [30, 50, 100, 250], 'do_sample': [False], 'temperature': [0.6, 0.7, 0.8], 'top_p': [0.9, 0.95, 1.]}
    for parm in [dict(zip(parm_grid.keys(), values)) for values in product(*parm_grid.values())]:
        temp = Main(df = 'churn_com_texto.csv').searchParametersMariTalk(**parm)
        temp['parm_grid'] = parm
        temp["confusion_matrix"] = confusion_matrix(temp['classificacao_original'], temp['classificacao_modelo'])
        temp["accuracy"] = accuracy_score(temp['classificacao_original'], temp['classificacao_modelo'])
        temp["recall"] = recall_score(temp['classificacao_original'], temp['classificacao_modelo'], average='weighted')
        temp["precision"] = precision_score(temp['classificacao_original'], temp['classificacao_modelo'], average='weighted')
        temp["f1_score"] = f1_score(temp['classificacao_original'], temp['classificacao_modelo'], average='weighted')
        parm_var.append(temp)
    
    best_parameters = max(parm_var, key=lambda x:x['accuracy'])
    print('best parameters: ', json.dumps(best_parameters['parm_grid'], indent=2))
    print('\n')
    print('confusion_matrix:\n', best_parameters.get('confusion_matrix'))
    print('\n')
    print('accuracy: ', best_parameters.get('accuracy'))
    print('recall: ', best_parameters.get('recall'))
    print('precision: ', best_parameters.get('precision'))
    print('f1_score: ', best_parameters.get('f1_score'))

    ## Classificação
    df = Main(df = 'churn_com_texto.csv')._clearWords()
    index_train = df.loc[:66].index.to_list()
    index_classifier=df.loc[67:].index.to_list()
    
    labels = Main(df = 'churn_com_texto.csv', index_train=index_train, index_classifier=index_classifier).mariTalkClassifier(**best_parameters['parm_grid'])
    print('labels: ', labels['classificacao_modelo'])
    df.loc[index_classifier, 'Número de Reclamações'] = labels['classificacao_modelo']
    df.to_csv(os.path.join(Path().absolute()) + f"\\data\\churn_tratado_classificado.csv", index=False)

