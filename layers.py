import tensorflow as tf
import numpy as np
import h5py, random, csv, gzip, time 
import pickle
from util import *
import os
class RMN(object):
	def __init__(self, d_word, d_char, d_book, d_hidden, len_voc, num_descs, num_chars, num_books, span_size, We, freeze_words=True, eps=1e-5, lr=0.01, negs=10 ):
		self.d_word = d_word
		self.d_char = d_char
		self.d_book = d_book
		self.d_hidden = d_hidden
		self.len_voc = len_voc
		self.num_descs = num_descs
		self.num_chars = num_chars
		self.num_books = num_books
		self.span_size = span_size
		self.We = We
		
		self.freeze_words = freeze_words
		self.eps = eps
		self.lr = lr
		self.num_negs = negs
		self.previous_state = tf.Variable(tf.zeros([1,num_descs]), trainable=True, name="previous_state") 
		
		self.input_spans = tf.placeholder(tf.int32, [None, self.span_size], name="input_spans")

		self.input_neg = tf.placeholder(tf.int32, [self.num_negs, self.span_size], name="input_neg")

		self.input_chars = tf.placeholder(tf.int32, [2, ], name="input_chars")#등장인물들

		self.input_book = tf.placeholder(tf.int32, [1, ], name="input_book")

		self.input_currmask = tf.placeholder(tf.float32, [None, self.span_size], name="input_currmask")

		self.input_dropmask = tf.placeholder(tf.float32, [None, self.span_size], name="input_dropmask")

		self.input_negmask = tf.placeholder(tf.float32, [self.num_negs, self.span_size], name="input_negmask")
		
		self.length = 5
		
		print ( self.input_spans,'\n', self.input_neg,'\n', self.input_chars,'\n', self.input_book,'\n', self.input_currmask,'\n', self.input_dropmask,'\n', self.input_negmask)

		self.loss,self.desciptor_R, self.input_recu = self._RMN_NETWORK()
        
	def EmbeddingLayer(self, E_w, input_spans, train=False):
		self.W = tf.Variable( E_w,name="Embedding_W",trainable=train)
		embedding_spans = tf.nn.embedding_lookup(self.W,input_spans)
		return embedding_spans
		
	def AverageLayer(self, emb , mask, d_word=300):
		emb_average = tf.reduce_sum(emb * mask[:, :, None], 1,keep_dims=True)/tf.reduce_sum(mask, 1,keep_dims=True)[:,None]
		#print (tf.reshape(emb_average,[-1,d_word]))
		return tf.reshape(emb_average,[-1,d_word])
		
	def ConcatLayer(self, inputs, input_d, chars_d ,books_d, **kwargs):
		self.W_i = tf.get_variable("W_i", shape=[input_d, input_d], initializer=tf.contrib.layers.xavier_initializer())
		self.b_i = tf.Variable(tf.constant(0.1, shape=[input_d,]), name="b_i")
		self.W_c = tf.get_variable("W_c", shape=[input_d, chars_d], initializer=tf.contrib.layers.xavier_initializer())
		self.W_b = tf.get_variable("W_b", shape=[input_d, books_d], initializer=tf.contrib.layers.xavier_initializer())
		
		input_vec = tf.nn.xw_plus_b(inputs[0], self.W_i, self.b_i, name="input_vec")
		char_vec = tf.matmul(self.W_c,tf.transpose(tf.reduce_sum(inputs[1],axis=0,keep_dims=True)),name="char_vec")
		book_vec = tf.matmul(self.W_b,tf.transpose(tf.reshape(inputs[2],[-1,books_d])),name="book_vec")
		#print (tf.nn.relu(input_vec+tf.transpose(char_vec)+tf.transpose(book_vec)))
		return tf.nn.relu(input_vec+tf.transpose(char_vec)+tf.transpose(book_vec),name="concat_relu")
		
	def RecurrentRelationshipLayer(self, concat, d_word, d_hidden, num_descs):
		self.W_C = tf.get_variable("W_C", shape=[d_word, num_descs], initializer=tf.contrib.layers.xavier_initializer())
		self.W_P = tf.get_variable("W_P", shape=[num_descs, num_descs], initializer=tf.contrib.layers.xavier_initializer())
		alpha = tf.constant(0.5,name='alpha')
		hidden_size = num_descs
		cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
		outputs, states = tf.nn.static_rnn(cell, tf.transpose(concat), dtype=tf.float32) 
		#
		#    수정 사항
		#    rnn을 쓸거면 멘션 통채로 넘겨야함! 지금은 멘션 하나씩 넘기는 형태
		#    => 고로 main.py에서 부터 수정해야함
		#	 우여 수정하길 
		#
		# state = self.previous_state = (1-alpha) * tf.nn.softmax(tf.matmul(concat, self.W_C)+tf.matmul(self.previous_state, self.W_P)) + (alpha)*self.previous_state
		# previous_state  = tf.zeros([1, num_descs],name='previous_state') 
		# states=list()
		# num_state=0
		# in_concat=tf.unpack(concat)
		# for row in range(self.length):
			# state = previous_state = alpha*tf.nn.softmax(tf.matmul(tf.expand_dims(in_concat[0],0),W_C)+tf.matmul(previous_state,W_P))+(1-alpha)*previous_state
			# states.append(state)
			# num_state+=1
		# state=tf.pack(states)
		return outputs, states
		
	def ReconstructionLayer(self, recu,num_descs,d_word):
		self.W_R = tf.get_variable("R_desciptor", shape=[num_descs, d_word], initializer=tf.contrib.layers.xavier_initializer())
		recon = tf.matmul(recu,self.W_R,name='recon')
		#print( recon )
		return recon, self.W_R
		
	def hinge_loss(self, emb, neg, recon):
		Hinge_loss = tf.reduce_sum(1.-tf.reduce_sum( recon*emb, 1,keep_dims=True)+tf.matmul(recon,tf.transpose(neg)), 1, keep_dims=True)
		return Hinge_loss
		
	def penalty(self, desciptor_R, eps):
		norm_R = tf.nn.l2_normalize(desciptor_R,1)
		ortho_penalty = eps * tf.reduce_sum((tf.matmul(norm_R,norm_R,transpose_b=True) - tf.eye(self.num_descs))**2)
		return ortho_penalty
		
	def _RMN_NETWORK(self):
	# negative examples should use same embedding matrix

		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			embedding_spans = self.EmbeddingLayer(self.We, self.input_spans)
			embedding_neg = self.EmbeddingLayer(self.We, self.input_neg)
			embedding_chars = self.EmbeddingLayer(tf.random_uniform([self.num_chars, self.d_char], -1.0, 1.0), self.input_chars)
			embedding_books = self.EmbeddingLayer(tf.random_uniform([self.num_books, self.d_book], -1.0, 1.0), self.input_book)
			
		with tf.device('/gpu:0'), tf.name_scope("AverageLayer" ):
			emb_average = self.AverageLayer(embedding_spans, self.input_currmask, self.d_word)
			drop_average = self.AverageLayer(embedding_spans, self.input_dropmask ,self.d_word)
			neg_average = self.AverageLayer(embedding_neg, self.input_negmask, self.d_word)
			
		with tf.device('/gpu:0'), tf.name_scope("ConcatLayer" ):
			input_concat = self.ConcatLayer([drop_average,embedding_chars,embedding_books], self.d_word, self.d_char, self.d_book)
		with tf.device('/gpu:0'), tf.name_scope("RecurrnetRelationshipLayer" ):
			input_recu, p_s = self.RecurrentRelationshipLayer(input_concat, self.d_word, self.d_hidden, self.num_descs)
		
		with tf.device('/gpu:0'), tf.name_scope("ReconstructionLayer"):
			input_recon, desciptor_R = self.ReconstructionLayer(input_recu, self.num_descs, self.d_word)
			#print("recon",input_recon)
		with tf.device('/gpu:0'), tf.name_scope("l2_normalize"):
			emb = tf.nn.l2_normalize(emb_average,1)
			neg = tf.nn.l2_normalize(neg_average,1)
			recon = tf.nn.l2_normalize(input_recon,1)
			print(emb,'\n',neg,'\n',recon)
		with tf.device('/gpu:0'), tf.name_scope("loss"):
			loss = self.hinge_loss(emb, neg, recon)
			loss += self.penalty(desciptor_R, self.eps)
			print (loss)
		return loss, desciptor_R, input_recu
		