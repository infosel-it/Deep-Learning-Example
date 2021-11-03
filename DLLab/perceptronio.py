import numpy as np 

class Perceptron(object):
   
    def __init__(self, input_size, weights,lr):
        self.W = weights
        # add one for bias
#        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
 
    def predict(self, x,theta):
#        print(" theta :", theta)
        z = self.W.T.dot(x)-(theta)
        print("X, Z :", x, z)
        z = round(z,2)
        a = self.activation_fn(z)
       
        return a
    
    def fit(self, X, d,theta,epochs):
        count=1
        for _ in range(epochs):
            
            print(" ==== Epoch: ====", count)
            count = count+1
            for i in range(d.shape[0]):
                x = X[i]
                print("input", x , "\t", "Weight:",self.W )
                y = self.predict(x,theta)
                e = d[i] - y
                print("desired:",d[i],"estimate:",y,"error = :",e)
                print("Lamda W1: ", self.lr * e * x[0] )
                print("Lamda W2: ", self.lr * e * x[1] )
                self.W = self.W + (self.lr * e * x)
                print("Self.W After Lamda Add", self.W )
                print("Output x", x , "\t", "Weight:",self.W )
                print()

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

    logicGate = input("Enter Logic Gate to test 'or','and','nor','nand':")
    w1 = float(input("Enter Weights1: ")) 
    w2 = float(input("Enter Weights2: ")) 
    bias = float(input("Enter Bias Value: ")) 
    learning_rate = float(input("Enter Learning Rate: ")) 
    epochs = int(input("Enter Max Epochs Value: ")) 
    
    weights = np.array([w1,w2])
    perceptron = Perceptron(2,weights,learning_rate)
    
    if logicGate == "and":
    # Fit for AND Gate     
        print(" And gate...")
        perceptron.fit(X, dand,bias, epochs)
        print(perceptron.W)
    
    elif logicGate == "or":
    # Fit for AND Gate     
        print(" Or gate...")
        perceptron.fit(X, dor,bias, epochs)
        print(perceptron.W)
        
    elif logicGate == "nor":
    # Fit for AND Gate     
        print(" Nor gate...")
        perceptron.fit(X, dnor,bias, epochs)
        print(perceptron.W)
        
    elif logicGate == "nand":
    # Fit for AND Gate     
        print(" Nand gate...")
        perceptron.fit(X, dnand,bias, epochs)
        print(perceptron.W)                    
    else:
        print("Unknown Input ...")