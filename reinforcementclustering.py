
from kmeans import *

import copy



class ReinforcementClustering:

	def __init__(self,kmeans_iterations,reinforcement_iterations,doc_clusters,num_doc_cluster,num_word_cluster):

		self.kmeans_iterations = kmeans_iterations
		self.reinforcement_iterations = reinforcement_iterations
		self.doc_cluster = doc_clusters
		self.word_cluster = np.transpose(doc_clusters)
		self.num_doc_cluster = num_doc_cluster
		self.num_word_cluster = num_word_cluster

		

	def do_reinforcement(self):
		'''
		Implements reinforcement clustering by alternating between word and doc clusters for given number of iterations.
		'''

		wmap = copy.deepcopy(self.word_cluster)
		
		for i in range(self.reinforcement_iterations):
			print "reinforcement iteration",i

			W= Kmeans(wmap,self.kmeans_iterations,self.num_word_cluster)
			cid_w,cs_w  = W.do_kmeans()
			flag = True
			dmap =  self.reconstruct(flag,cid_w)
			
			D =   Kmeans(dmap,self.kmeans_iterations,self.num_doc_cluster)
			cid_d,cs_d  = D.do_kmeans()
			flag = False
			wmap = self.reconstruct(flag,cid_d)

		return cid_d,cs_d,cs_w



	def reconstruct(self,flag,cid):

		'''
		Reconstructs a new document matrix based on clustering of words (and vice-versa)
		'''

		if flag:

			orig_map =  self.doc_cluster
			current_map = self.word_cluster
			current_clusters = self.num_word_cluster

		else:

			orig_map = self.word_cluster
			current_map = self.doc_cluster
			current_clusters = self.num_word_cluster
		
		num_docs = orig_map.shape[0]
		trans_current_map=np.transpose(current_map)
		new = np.zeros((current_clusters,num_docs))
		range_clusters = np.arange(current_clusters)

		for i in range(current_clusters):
			new[i,:] = sum(current_map[cid == i,:])
		dmap = np.transpose(new)
		
		return dmap


		

				
