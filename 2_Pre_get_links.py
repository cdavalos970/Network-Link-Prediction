from random import randint
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
import networkx as nx
import math

file_links = "/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/train.txt"
file_output = "/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis.csv"
file_output_f = "/Volumes/Transcend/GitHub/Statistical Machine Learning/Assignment_1/dataFrame_analysis_f_1.csv"
G = nx.Graph()

num_nodes = 4084
beta = 0.05

G.add_nodes_from(range(0,num_nodes))

df_author = pd.read_csv('authorMatrix.csv')  

dictLink = {}

def neighbors(dict1, dict2):
    setA = set(dict1)
    setB = set(dict2)

    return len(setA.intersection(setB))

def pref_attach(dict1, dict2):
    setA = len(set(dict1)) - 1
    setB = len(set(dict2)) - 1

    return setA*setB

def katz_function(nodeA, nodeB):
    dictNode = dict()
    paths = nx.all_simple_paths(G, source=nodeA, target=nodeB, cutoff=4)
    for path in paths:
        len_path = len(path) - 1
        if len_path > 1:
            if len_path not in dictNode.keys():
                dictNode[len_path] = 1
            else:
                dictNode[len_path] = dictNode[len_path] + 1
    
    score = 0
    for key, value in dictNode.items():
        score = score + (beta**key)*value
    return score

'''

dfTrue = pd.DataFrame(columns=['Source', 'Sink'])
with open(file, "r") as f:
    for line in f:
        splitLine = line.split()
        nodeOne = int(splitLine[0])
        for link in range(1, len(splitLine)):
            nodeTwo = int(splitLine[link])
            if nodeOne < nodeTwo:
                dfNew = pd.DataFrame({'Source': [nodeOne], 'Sink': [nodeTwo]})
                dfTrue = dfTrue.append(dfNew, ignore_index = True)
            else:
                dfNew = pd.DataFrame({'Source': [nodeTwo], 'Sink': [nodeOne]})
                dfTrue = dfTrue.append(dfNew, ignore_index = True)        

dfTrue.to_csv(r'authorsLinks.csv', index = False, header=True)
'''

count = 0
with open(file_links, "r") as f:
    for line in f:
        splitLine = line.split()
        nodeOne = int(splitLine[0])
        dictLink.setdefault(nodeOne, {})
        for link in range(1, len(splitLine)):
            nodeTwo = int(splitLine[link])
            G.add_edge(nodeOne, nodeTwo)
            count = count + 1
            dictLink.get(nodeOne).setdefault(nodeTwo, 1)
count = count/2

while count > 0:
    num = np.random.choice(num_nodes, 2, replace = False)
    nodeOne = num[0]
    nodeTwo = num[1]
    if dictLink.get(nodeOne, 2) == 2:
        dictLink.setdefault(nodeOne, {})
    if dictLink.get(nodeTwo, 2) == 2:
        dictLink.setdefault(nodeTwo, {})
    if dictLink.get(nodeOne).get(nodeTwo, 2) != 1:
        dictLink.get(nodeOne).setdefault(nodeTwo, 0)
        dictLink.get(nodeTwo).setdefault(nodeOne, 0)
        count = count - 1

df_analysis = pd.DataFrame(columns=['Source', 'Sink', 'Jac_Score_Key', 'Jac_Score_Venue', 'neighbors', 'short', 'path_num', 'dif_papers', 'pref', 'adamic', 'sim', 'katz', 'dif_rate', 'dif_last', 'jaccard', 'Node'])

for keyNode, valueNode in dictLink.items():
    df_analysisCur = pd.DataFrame(columns=['Source', 'Sink', 'Jac_Score_Key', 'Jac_Score_Venue', 'neighbors', 'short', 'path_num', 'dif_papers', 'pref', 'adamic', 'sim', 'katz', 'dif_rate', 'dif_last', 'jaccard', 'Node'])
    if keyNode == 0:
        df_analysisCur.to_csv(file_output, index = False, header=True)
    for keyLink, valueLink in valueNode.items():
        keyNode = int(keyNode)
        keyLink = int(keyLink)
        if keyNode < keyLink:
            dfKeyNode = df_author[keyNode:(keyNode + 1)]
            dfKeyLink = df_author[keyLink:(keyLink + 1)]

            dfKeyNode_key = dfKeyNode.filter(regex=r'^key',axis=1).values.tolist()[0]
            dfKeyLink_key = dfKeyLink.filter(regex=r'^key',axis=1).values.tolist()[0]
            j_score_keyword = jaccard_score(dfKeyNode_key, dfKeyLink_key)

            dfKeyNode_venue = dfKeyNode.filter(regex=r'^venue',axis=1).values.tolist()[0]
            dfKeyLink_venue = dfKeyLink.filter(regex=r'^venue',axis=1).values.tolist()[0]
            j_score_venue = jaccard_score(dfKeyNode_venue, dfKeyLink_venue)

            numNeighbors_1 = neighbors(dictLink.get(keyNode), dictLink.get(keyLink))
            katz_score = katz_function(keyNode, keyLink)

            jaccard = ''
            shortest_path = ''
            pa = 0
            adamic = ''
            ra = ''
            path_num = 0
            sim = 0
            dif_rate = math.inf
            '''
            try:
                shortest_path = nx.shortest_path_length(G, source=keyNode, target = keyLink)
            except Exception:
                pass
            '''
            try:
                pa_vector = nx.jaccard_coefficient(G, [(keyNode, keyLink)])
                for u, v, p in pa_vector:
                    jaccard = round(p,6)
            except Exception:
                pass
            '''
            try:
                pa_vector = nx.preferential_attachment(G, [(keyNode, keyLink)])
                for u, v, p in pa_vector:
                    pa = p
            except Exception:
                pass
            '''
            pa = pref_attach(dictLink.get(keyNode), dictLink.get(keyLink))
            
            try:
                adamic_vector = nx.adamic_adar_index(G, [(keyNode, keyLink)])
                for u, v, p in adamic_vector:
                    adamic = round(p,6)
            except Exception:
                pass
            
            try:
                ra_vector = nx.resource_allocation_index(G, [(keyNode, keyLink)])
                for u, v, p in ra_vector:
                    ra = round(p,6)
            except Exception:
                pass
            '''
            try:
                path_num = len(list(nx.all_simple_paths(G, source=keyNode, target=keyLink)))
            except Exception:
                pass
            
            try:
                sim = nx.simrank_similarity(G, source=keyNode, target=keyLink)
            except Exception:
                pass
            '''
            dif_papers = round(abs(dfKeyNode['num_papers'][keyNode]/dfKeyLink['num_papers'][keyLink]),6)

            try:
                dif_rate = round((dfKeyNode['num_papers'][keyNode]/(dfKeyNode['first'][keyNode] - dfKeyNode['last'][keyNode]))/(dfKeyLink['num_papers'][keyLink]/(dfKeyLink['first'][keyLink] - dfKeyLink['last'][keyLink])),6)
            except Exception:
                pass

            dif_last = abs(dfKeyNode['last'][keyNode] - dfKeyLink['last'][keyLink])

            dfNew = pd.DataFrame({'Source': [keyNode], 'Sink': [keyLink], 'Jac_Score_Key': round(j_score_keyword,6), 'Jac_Score_Venue': round(j_score_venue,6), 'neighbors': numNeighbors_1, 'short': shortest_path, 'path_num': path_num, 'dif_papers': dif_papers, 'pref': pa, 'adamic': adamic, 'resour': ra, 'sim': sim, 'katz':katz_score, 'dif_rate':dif_rate, 'dif_last':dif_last, 'jaccard':jaccard, 'Node': [valueLink]})
            df_analysisCur = df_analysisCur.append(dfNew, ignore_index = True)
    print(keyNode)
    df_analysisCur.to_csv(file_output, index = False, header=False, mode='a')
    df_analysis = df_analysis.append(df_analysisCur, ignore_index = True)

df_analysis.to_csv(file_output_f, index = False, header=True)