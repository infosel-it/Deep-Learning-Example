import numpy as np 

class Perceptron(object):
   
    def __init__(self, input_size, lr=0.2):
        self.W = np.array([0.3,-0.2])
        # add one for bias
#        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
 
    def predict(self, x,theta):
        z = self.W.T.dot(x)-theta
        z = round(z,2)
        a = self.activation_fn(z)
       
        return a
    
    def fit(self, X, d,theta,epochs):
        count=1
        for _ in range(epochs):
            
            print("Epoch: ", count)
            count = count+1
            for i in range(d.shape[0]):
                x = X[i]
                print("input", x , "\t", "Weight:",self.W )
                y = self.predict(x,theta)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
                

if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    dand = np.array([0, 0, 0, 1])
    dor = np.array([0, 1, 1, 1])
    
    dnor = np.array([1, 0, 0, 0])
    dnand = np.array([1, 1, 1, 0])    
     
    perceptron = Perceptron(input_size=2)
    theta=0.4
    epochs =10
  
# Fit for AND Gate     
    perceptron.fit(X, dand,theta, epochs)
    print(perceptron.W)
    
# Fit for OR Gate     
 #   perceptron.fit(X, dor,theta, epochs)
 #   print(perceptron.W)
    
# Fit for NAND Gate     
#    perceptron.fit(X, dnand,theta, epochs)
#    print(perceptron.W)
    
# Fit for NOR Gate     
#    perceptron.fit(X, dnor,theta, epochs)
#    print(perceptron.W)    
