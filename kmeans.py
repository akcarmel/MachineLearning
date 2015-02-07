
import random
import math
import numpy as np
import copy




class Kmeans:

	def __init__(self,given_map,num_iterations,K):

		self.num_iterations = num_iterations
		self.K = K
		self.map = given_map
		 
		self.mapc = self.normalize(copy.deepcopy(given_map))
		self.cosine_sum = 0.0

	def normalize(self,a):
		'''
		L2 normalization of matrix a. Each row is ONE data point.
		'''
		dr = np.sqrt(np.sum(a*a, axis=1))
		dr[dr==0,:] = 1.0
		f=a/dr[:,np.newaxis]
		return f
	
	def set_centroids(self):

		'''
		Sets centroids to be 'K' random data points.  Each row is ONE data point.
		'''

		rand_indices = random.sample(range(self.map.shape[0]),self.K)
		
		centroids = self.map[rand_indices,:]
		

		return centroids
	
	def expectation(self,centroids):
		'''
		Assigns a centroid to every data point.  Each row is ONE data point.
		'''
		
		centroids_norm = self.normalize(copy.deepcopy(centroids))
		CS = self.mapc.dot(np.transpose(centroids_norm))  #Cosine Matrix
		obj_to_cluster = np.argmax(CS,axis=1)
		row_idx = np.arange(CS.shape[0])
		col_idx = obj_to_cluster
		self.cosine_sum =  np.sum(CS[row_idx,col_idx])  #Only pick cosine values of assigned clusters.

		

		return obj_to_cluster


		

	def maximization(self,obj_to_cluster,centroids):
		'''
		Recalculates the Centroids
		'''

		counter = -1    #counter keeps a track of cluster number, this is especially useful if no data points have been assigned to a cluster.

		for i in range(self.K):
			z=obj_to_cluster == i
			
			if z.any():
				counter += 1
				centroids[counter,:] = np.mean(self.map[z,:],axis=0)
			else:         #This condition happens when no datapoint has been assigned to a cluster.
				continue

		return centroids

	def do_kmeans(self):
		'''
		Runs expectation, maximization one after the other
		'''

		centroids = self.set_centroids()
		prev_centroids = copy.deepcopy(centroids)
		
		for i in range(self.num_iterations):

			

			print "kmeans iteration",i

			obj_to_cluster=self.expectation(centroids)
			centroids=self.maximization(obj_to_cluster,centroids)
			if np.array_equal(centroids,prev_centroids):   #If centroids have not changed break out of the loop.
				obj_to_cluster=self.expectation(centroids)
				break
			prev_centroids = copy.deepcopy(centroids)
			



		return obj_to_cluster,self.cosine_sum