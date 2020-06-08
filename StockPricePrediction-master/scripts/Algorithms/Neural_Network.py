#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

'''
    Neural Network Implementation
'''
class NeuralNet():
    
    def __init__(self, num_nodes, weights_=[], classification=True, auto_encoder=False, penalty=0., learn_rate=0.01):
        self.num_nodes = num_nodes
        self.weights_ = weights_
        self.is_fit = False
        self.classification = classification
        self.auto_encoder = auto_encoder
        if auto_encoder:
            self.classification = False
        self.K = 0
        self.penalty = penalty
        self.learn_rate = learn_rate
        np.seterr(all='warn')
    
    def __str__(self):
        return "Neural Networks("

    def initWeights(self, X, nclass):
        bias = np.ones((X.shape[0],1))
        X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
        sizeX = X.shape[1]
        node_weights_ = np.random.uniform(-0.08,0.08,size=(self.num_nodes, sizeX)) #M+1 by P+1
        output_weights_ = np.random.uniform(-0.08,0.08,size=(nclass, self.num_nodes+1)) #K by M+1
        return X, [node_weights_, output_weights_] #nrows = n_nodes, ncols = sizeX
        
    def sigmoid(self, alpha_, X_):
        v_ = alpha_.dot(X_.T)
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
        return 1./(1+np.exp(-v_))
        
    def relu(self, alpha_, X_):
        try:
            v_ = alpha_.dot(X_.T)
        except ValueError:
            v_ = X_.dot(alpha_)
            v_ = v_.T
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
        return np.maximum(v_,np.zeros(v_.shape))
        
    def drelu(self, alpha_, X_):
        try:
            v_ = alpha_.dot(X_.T)
        except ValueError:
            v_ = X_.dot(alpha_)
            v_ = v_.T
        v_[v_ <= 0.] = 0
        v_[v_ > 0.] = 1.
        return v_
        
    def tanh(self, alpha_, X_):
        try:
            v_ = alpha_.dot(X_.T)
        except ValueError:
            v_ = X_.dot(alpha_)
            v_ = v_.T
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
        return np.tanh(v_)
        
    def dtanh(self, alpha_, X_):
        return 1 - np.multiply(self.tanh(alpha_, X_), self.tanh(alpha_, X_))
        
    def softmax(self, T):
        T[T < -300] = -300
        T[T > 300] = 300
        return (np.exp(T)/np.sum(np.exp(T), axis=0)).T #(K by N) / elementwise(1 by N)
        
    def initNodes(self, X, Y):
        K = self.K
        if self.weights_ == []:
            X, weights = self.initWeights(X, K)
        else:
            weights = self.weights_
            bias = np.ones((X.shape[0],1))
            X = np.hstack((bias,X))
        return X, weights
        
    def backPropagate(self, weights, next_weights, X, Y, rho, old_del_alpha, old_del_beta, _dropout, back_delta=0., fine_tune=False):
        """
        feed forward then back propagate error, update weights
        """
        learn_rate = self.learn_rate
#        beta = 6.
#        sparsity = 0.05
        if _dropout:
            drop = np.random.uniform(0, 1, weights[0].shape[0])
        if self.auto_encoder and not fine_tune:
            sig = self.relu(weights[0], X)
            dsig = self.drelu(weights[0], X)
            if _dropout:
                sig[drop>=0.5,:] = 0.
                dsig[drop>=0.5,:] = 0.
#            avg_sparsity = np.mean(sig, axis=1)
        elif not self.auto_encoder and not fine_tune:
            sig = self.tanh(weights[0], X)
            dsig = self.dtanh(weights[0], X)
            if _dropout:
                sig[drop>=0.5,:] = 0.
                dsig[drop>=0.5,:] = 0.
            
        if not fine_tune:
            bias = np.ones((1,sig.shape[1]))
            sig = np.vstack((bias,sig))
            hidden_out = weights[1].dot(sig)
            if self.classification:
                h = self.softmax(hidden_out)
                forward_error = h - Y
            else:
                h = hidden_out.T
                forward_error = h - Y[:,1:] #both N by K
            dRdBeta = sig.dot(forward_error)/forward_error.shape[0] #(M+1 by N)*(N by K) = M+1 by K gradient-force for each neuron

        if fine_tune:
            prop_back = np.multiply(back_delta.dot(next_weights[:,1:]),dsig.T)
            dRdAlpha = prop_back.T.dot(X)/X.shape[0]
        elif not fine_tune and not self.auto_encoder:
            back_error = np.multiply((forward_error.dot(weights[1][:,1:])),(dsig.T)) #((N by K)*(K by M+1))*ewise(N by M+1) = N by M+1
            prop_back = back_error
        elif not fine_tune and self.auto_encoder:
            back_error = np.multiply((forward_error.dot(weights[1][:,1:])),(dsig.T))
            prop_back = 0.
#            back_error = np.multiply((forward_error.dot(weights[1][:,1:])),(dsig.T)) + beta*(-sparsity/avg_sparsity+(1-sparsity)/(1-avg_sparsity))
        if not fine_tune:
            dRdAlpha = (back_error.T).dot(X)/X.shape[0]
            del_beta = rho*old_del_beta - learn_rate*dRdBeta.T
        else:
            del_beta = 0.
        del_alpha = rho*old_del_alpha - learn_rate*dRdAlpha
        
        """Bias weights do not get penalized"""
        if not fine_tune:
            bias1 = np.zeros((weights[1].shape[0], 1))
            weights[1] = weights[1] + del_beta + np.hstack((bias1,self.penalty*weights[1][:,1:])) #M+1 by K
        bias0 = np.zeros((weights[0].shape[0], 1))
        weights[0] = weights[0] + del_alpha + np.hstack((bias0,self.penalty*weights[0][:,1:])) #M+1 by P+1

        return weights, del_alpha, del_beta, prop_back
        
    def feedForward(self,X,layers):
        activations = []
        for i,layer in enumerate(layers):
            if i == 0:
                if layer.auto_encoder:
                    if layer._dropout:
                        sig = layer.relu(layer.weights_[0]/2.,X)
                    else:
                        sig = layer.relu(layer.weights_[0], X)
#                    sig = layer.tanh(layer.weights_[0], X)
                else:
                    if layer._dropout:
                        sig = layer.tanh(layer.weights_[0]/2.,X)
                    else:
                        sig = layer.tanh(layer.weights_[0], X)
            else:
                if layer.auto_encoder:
                    bias = np.ones((1,sig.shape[1]))
                    sig = np.vstack((bias,sig))
                    if layer._dropout:
                        sig = layer.relu(layer.weights_[0]/2., sig.T)
                    else:
                        sig = layer.relu(layer.weights_[0], sig.T)
                else:
                    if layer._dropout:
                        sig = layer.tanh(layer.weights_[0]/2., sig.T)
                    else:
                        sig = layer.tanh(layer.weights_[0], sig.T)
            activations.append(sig.T)
        return activations
            
    def fit(self, X, Y, rho=0., maxiter=300, tol=0.000001, anneal=False, t_0=50, dropout=False, batch=40, SGD=True, layers=[], fine_tune=False):
        self._dropout = dropout      
        grad_alpha, grad_beta = 0., 0.
        layer_alphas = [0. for i in range(len(layers))]
        layer_betas = layer_alphas
        self.is_fit = True
        if self.classification:
        #one-hot encode Y
            try:
                #if already one-hot encoded, pass Y as Y_new
                if Y.shape[1] > 1:
                    Y_new = Y
                    self.K = Y.shape[1]
                #else one-hot encode Y as Y_new
                else:
                    self.K = len(set(Y.flatten()))
                    Y_new = np.zeros((len(Y),self.K))
                    for i,v in enumerate(Y):
                        Y_new[i,v] = 1.
            #if Y.shape[1] null (1D array), one-hot encode it as Y_new
            except IndexError:
                self.K = len(set(Y.flatten())) #ditto
                Y_new = np.zeros((len(Y),self.K))
                for i,v in enumerate(Y):
                    Y_new[i,v] = 1.
        else:
            Y_new = Y
            if not self.auto_encoder:
                self.K = 1
            else:
                self.K = Y.shape[1]
        if layers == []:
            X, w = self.initNodes(X, Y_new)
        else:
            bias = np.ones((X.shape[0],1))
            X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
            X_ = self.feedForward(X,layers)
            X_[-1], w = self.initNodes(X_[-1], Y_new)

        for i in range(maxiter):
            if anneal and i != 0 and i % t_0 == 0:
                self.learn_rate /= (float(i)/t_0)
            if not SGD:
                if fine_tune and layers != []:
                    X_hidden = self.feedForward(X,layers)
                    bias = np.ones((X_hidden[-1].shape[0],1))
                    X_hidden[-1] = np.hstack((bias,X_hidden[-1]))
                
                    w, grad_alpha, grad_beta, back_error = self.backPropagate(w, 0., X_hidden[-1], Y_new, rho, grad_alpha, grad_beta, dropout, back_delta=0., fine_tune=False)
                    next_weights = w[0]                    
                    for i,layer in enumerate(layers[::-1]):
                        if len(layers)-i-2 >= 0:
                            activations = X_hidden[len(layers)-i-2]
                            bias = np.ones((activations.shape[0], 1))
                            activations = np.hstack((bias,activations))
                        else:
                            activations = X
                        layer.weights_, layer_alphas[i], layer_betas[i], back_error = layer.backPropagate(layer.weights_, next_weights, activations, Y_new, rho, layer_alphas[i], layer_betas[i], dropout, back_delta=back_error, fine_tune=True)                     
                        next_weights = layer.weights_[0]
                elif not fine_tune:
                    if self.auto_encoder:
                        choose = np.random.binomial(1, 0.9, size=X.shape)
                        X_noisy = np.multiply(choose, X)
                    else:
                        X_noisy = X
                    w, grad_alpha, grad_beta, back_error = self.backPropagate(w, 0., X_noisy, Y_new, rho, grad_alpha, grad_beta, dropout, back_delta=0., fine_tune=fine_tune)

            else:
                samples = np.random.choice(range(len(X)),size=batch,replace=False)
                if fine_tune and layers != []:
                    try:
                        X_hidden = self.feedForward(X[samples,:],layers)
                    except TypeError:
                        X_samples = [X[z] for z in samples]
                        X_hidden = self.feedForward(X_samples,layers)
                    bias = np.ones((X_hidden[-1].shape[0],1))
                    X_hidden[-1] = np.hstack((bias,X_hidden[-1]))
                
                    w, grad_alpha, grad_beta, back_error = self.backPropagate(w, 0., X_hidden[-1], Y_new[samples,:], rho, grad_alpha, grad_beta, dropout, back_delta=0., fine_tune=False)
                    next_weights = w[0]                    
                    for i,layer in enumerate(layers[::-1]):
                        if len(layers)-i-2 >= 0:
                            activations = X_hidden[len(layers)-i-2]
                            bias = np.ones((activations.shape[0],1))
                            activations = np.hstack((bias,activations))
                        else:
                            try:
                                activations = X[samples,:]
                            except TypeError:
                                activations = [X[z] for z in samples]
                        layer.weights_, layer_alphas[i], layer_betas[i], back_error = layer.backPropagate(layer.weights_, next_weights, activations, Y_new[samples,:], rho, layer_alphas[i], layer_betas[i], dropout, back_delta=back_error, fine_tune=True)                     
                        next_weights = layer.weights_[0]
                elif not fine_tune:
                    if self.auto_encoder:
                        choose = np.random.binomial(1,0.9,size=X[samples,:].shape)
                        X_noisy = np.multiply(choose, X[samples,:])
                        Y_test = X[samples,:]
                    else:
                        X_noisy = X[samples,:]
                        if self.classification:
                            Y_test = Y_new[samples,:]
                        else:
                            Y_test = Y_new[samples]
                    w, grad_alpha, grad_beta, back_error = self.backPropagate(w, 0., X_noisy, Y_test, rho, grad_alpha, grad_beta, dropout, back_delta=0., fine_tune=fine_tune)
            
        self.weights_ = w
        return layers
            
    def predict(self, X, proba=True, layers=[]):
        if self.is_fit:
            self.predictions = []
            if layers==[]:
                bias = np.ones((X.shape[0],1))
                X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
            if layers == []:
                if self.auto_encoder:
                    if self._dropout:
                        activation = self.relu(self.weights_[0]/2, X)
                    else:
                        activation = self.relu(self.weights_[0], X)
                else:
                    if self._dropout:
                        activation = self.tanh(self.weights_[0]/2., X)
                    else:
                        activation = self.tanh(self.weights_[0], X)
                bias = np.ones((1,activation.shape[1]))
                activation = np.vstack((bias,activation))
                response = self.weights_[1].dot(activation)
            else:
                activation = self.feedForward(X,layers)
                activation = activation[-1]
#            print activation.shape
#            if layers != []:
                bias = np.ones((activation.shape[0],1))
                activation = np.hstack((bias,activation))
                if self._dropout:
                    response = self.tanh(self.weights_[0]/2., activation)
                else:
                    response = self.tanh(self.weights_[0], activation)
                bias = np.ones((1,response.shape[1]))
                response = np.vstack((bias,response))
                response = self.weights_[1].dot(response)
            if self.classification:
                predictions = self.softmax(response)
                if not proba:
                    predictions = np.argmax(predictions, axis=1)
            else:
                predictions = response
            self.predictions = predictions
            return self.predictions
        else:
            return "Cannot predict without fitting data first!!"
    
    def hidden_activations(self, X):
        if self.is_fit:
            bias = np.ones((X.shape[0],1))
            X = np.hstack((bias,X))
            if self.auto_encoder:
                if self._dropout:
                    activations = self.relu(self.weights_[0]/2., X)
                else:
                    activations = self.relu(self.weights_[0], X)
#                activations = self.tanh(self.weights_[0], X)
            else:
                if self._dropout:
                    activations = self.tanh(self.weights_[0]/2., X)
                else:
                    activations = self.tanh(self.weights_[0], X)
            return activations.T
        else:
            return "Method 'hidden_activations' can only be called for auto encoders"
        
    def score(self, X_test, Y_test, layers=[]):
        predictions = self.predict(X_test, proba=False, layers=layers)
        if self.classification:
            try:
                if Y_test.shape[1] > 1:
                    num_correct = predictions == np.argmax(Y_test, axis=1)
                    return float(len(Y_test[num_correct]))/len(Y_test)
                else:
                    num_correct = predictions == np.array(Y_test).flatten()
                    return float(len(Y_test.flatten()[num_correct]))/len(Y_test)
            except IndexError:
                num_correct = predictions == np.array(Y_test).flatten()
                return float(len(Y_test.flatten()[num_correct]))/len(Y_test)
        else:
            n = len(Y_test)
            diff = predictions.T - Y_test
            MSE = 1. - sum(np.multiply(diff,diff))/n
            return MSE