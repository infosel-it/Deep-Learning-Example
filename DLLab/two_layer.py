class Twolayer_network:
  
  def __init__(self):
    self.w1 = np.random.randn()
    self.w2 = np.random.randn()
    self.w3 = np.random.randn()
    #self.w4 = np.random.randn()
    #self.w5 = np.random.randn()
    self.w6 = np.random.randn()
    self.w7 = np.random.randn()
    self.w8 = np.random.randn()
    self.w9 = np.random.randn()
    self.w10 = np.random.randn()
    self.w11 = np.random.randn()
    self.w12 = np.random.randn()
    self.b1 = 0
    self.b2 = 0
    self.b3 = 0
    self.b4 = 0
    self.b5 = 0
    
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def forward_pass(self, x):
    #input layer
    self.x1, self.x2 = x
    #hidden 1 
    self.a1 = self.w1*self.x1 + self.w2*self.x2 + self.b1
    self.h1 = self.sigmoid(self.a1)
    self.a2 = self.w3*self.x1 + self.w6*self.x2 + self.b2
    self.h2 = self.sigmoid(self.a2)

    #hidden 2 

    self.a3 = self.w7*self.h1 + self.w8*self.h2 + self.b3
    self.h3 = self.sigmoid(self.a3)
    self.a4 = self.w9*self.h1 + self.w10*self.h2 + self.b4
    self.h4 = self.sigmoid(self.a4)

    #output layer 
    self.a5 = self.w11*self.h3 + self.w12*self.h4 + self.b5
    self.h5 = self.sigmoid(self.a5)
    return self.h5
  
  def grad(self, x, y):
    self.forward_pass(x)  
    #self.dw1 = (self.h3-y) * self.h3*(1-self.h3) * self.w5 * self.h1*(1-self.h1) * self.x1
    #output layer
    self.dw11 = (self.h5-y) * self.h3 
    self.dw12 = (self.h5-y) * self.h4
    self.db5 = (self.h5-y)
    # hidden layer 2

    self.dh3 = (self.h5-y)*self.w11 
    self.dw7 = self.dh3 * self.h3*(1-self.h3) * self.h1
    self.dw8 = self.dh3 * self.h3*(1-self.h3) * self.h2
    self.db3 = self.dh3 * self.h3*(1-self.h3)

    self.dh4 = (self.h5-y)*self.w12 
    self.dw9 = self.dh4 * self.h4*(1-self.h4) * self.h1
    self.dw10 = self.dh4 * self.h4*(1-self.h4) * self.h2
    self.db4 = self.dh4 * self.h4*(1-self.h4)
                    
    #hidden layer 1(doubt)

    self.dh1 = self.dh3*self.w7 + self.dh4*self.w9 
    self.dw1 = self.dh1 * self.h1*(1-self.h1) * self.x1
    self.dw2 = self.dh1 * self.h1*(1-self.h1) * self.x2
    self.db1 = self.dh1 * self.h1*(1-self.h1)
     

    self.dh2 = self.dh3*self.w8 + self.dh4*self.w10 
    self.dw3 = self.dh2 * self.h2*(1-self.h2) * self.x1
    self.dw6 = self.dh2 * self.h2*(1-self.h2) * self.x2
    self.db2 = self.dh2 * self.h2*(1-self.h2)
    
  
  def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False, display_weight=False):
      
    if display_loss:
      loss = []
      #w1 = []
    
    for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
      dw1, dw2, dw3,dw6, dw7, dw8, dw9, dw10, dw11, dw12, db1, db2, db3, db4, db5  = [0]*15
      for x, y in zip(X, Y):
        self.grad(x, y)
        dw1 += self.dw1
        dw2 += self.dw2
        dw3 += self.dw3
        
        dw6 += self.dw6
        dw7 += self.dw7
        dw8 += self.dw8
        dw9 += self.dw9
        dw10 += self.dw10
        dw11 += self.dw11
        dw12 += self.dw12
        db1 += self.db1
        db2 += self.db2
        db3 += self.db3
        db4 += self.db4
        db5 += self.db5
        #db6 += self.db6
        
      m = X.shape[0]
      self.w1 -= learning_rate * dw1 / m
      self.w2 -= learning_rate * dw2 / m
      self.w3 -= learning_rate * dw3 / m
      #self.w4 -= learning_rate * dw4 / m
      #self.w5 -= learning_rate * dw5 / m
      self.w6 -= learning_rate * dw6 / m
      self.w7 -= learning_rate * dw7 / m
      self.w8 -= learning_rate * dw8 / m
      self.w9 -= learning_rate * dw9 / m
      self.w10 -= learning_rate * dw10 / m
      self.w11 -= learning_rate * dw11 / m
      self.w12 -= learning_rate * dw12 / m
      self.b1 -= learning_rate * db1 / m
      self.b2 -= learning_rate * db2 / m
      self.b3 -= learning_rate * db3 / m
      self.b4 -= learning_rate * db4 / m
      self.b5 -= learning_rate * db5 / m
      #self.b6 -= learning_rate * db6 / m
      
      
      if display_loss:
        #w1.append(self.w1)
        Y_pred = self.predict(X)
        loss.append(mean_squared_error(Y_pred, Y))
        #loss[i] = mean_squared_error(Y_pred, Y)
      
      if display_weight:
        weight_matrix = np.array([[self.b3,  self.w6, 
                                   self.b4, self.w7, self.w8, 
                                   self.b5, self.w9, self.w10, 
                                    self.w11, self.w12], 
                                  [0, 0, 0,
                                   self.b1, self.w1, self.w2,
                                   self.b2, self.w3, 
                                   0, 0, 0]])
        weight_matrices.append(weight_matrix)
    
    if display_loss:
      plt.plot(loss)
      plt.xlabel('Epochs')
      plt.ylabel('Log Loss')
      plt.show()  
  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.forward_pass(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred)
ffnw1 = Twolayer_network()
ffnw1.fit(X_train, Y_train, epochs=50, learning_rate=1, display_loss=True, display_weight=True)