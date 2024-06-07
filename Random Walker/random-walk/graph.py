import networkx as nx 
import walker 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import diags
from sklearn.preprocessing import normalize

import os 
import pickle

word_path = "data/alpha_0.0_s2v/vocab.csv"
semantic_path = "data/alpha_0.0_s2v/semantic_matrix.csv"
graph_path = "graph/semantic_graph.gpickle"


class semantic_graph():
    
    def __init__(self, word_path, semantic_path, graph_path):
        self.word_path = word_path
        
        self.semantic_path = semantic_path
        self.graph_path = graph_path
        
        
        self.words = pd.read_csv(word_path)["Word"].tolist()
        self.semantic = pd.read_csv(semantic_path, header=None, names=list(range(463)))
        # self.graph = nx.read_gpickle(graph_path)
        self.walker = walker
        
    def create_graph(self): 
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
        else: 
            word1 = [] 
            word2 = [] 
            similarity = []
            
            for i in range(len(self.words)):
                j = i 
                while j < len(self.words):
                    word1.append(self.words[i])
                    word2.append(self.words[j])
                    similarity.append(self.semantic.iloc[i, j])
                    j += 1
        
            self.df_graph = pd.DataFrame({"word1": word1, "word2": word2, "similarity": similarity})
            self.graph = nx.from_pandas_edgelist(self.df_graph, 'word1', 'word2', ['similarity'], create_using = nx.Graph)
            
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
    
    def visualize_graph(self): 
        nx.draw(self.graph)
        plt.savefig("graph/Graph.png", format="PNG")

    
class utils(): 
    def _weight_node(node, G, m, sub_sampling):
        z = G.degree(node, weight="weight") + 1
        weight = 1 / (z**sub_sampling)
        return weight

    def get_normalized_adjacency(G, sub_sampling=0.1):
        A = nx.adjacency_matrix(G).astype(np.float32)
        if sub_sampling != 0:
            m = len(G.edges)
            D_inv = diags([
                utils._weight_node(node, G, m, sub_sampling)
                for node in G.nodes
            ])
            A = A.dot(D_inv)

        normalize(A, norm="l1", axis=1, copy=False)
        return A


            
''' Run for example '''
# graph = semantic_graph(word_path, semantic_path, graph_path)
# graph.create_graph()
# graph.visualize_graph()