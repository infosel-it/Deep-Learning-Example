import numpy as np
from ordered_set import OrderedSet

class DataReader:
    def __init__(self, path, seq_length):
        
        self.fp = open(path, "r")
        self.data = self.fp.read()
        #find unique chars
        chars = list(OrderedSet(self.data))
        #create dictionary mapping for each char
        self.char_to_ix = {ch:i for (i,ch) in enumerate(chars)}
        self.ix_to_char = {i:ch for (i,ch) in enumerate(chars)}
        
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()

class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        # hyper parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.log = False
        # model parameters
        self.U = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_size, vocab_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (vocab_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))
        self.b = np.zeros((hidden_size, 1)) # bias for hidden layer
        self.c = np.zeros((vocab_size, 1)) # bias for output
        
        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)
        

    def softmax(self, x):
        p = np.exp(x- np.max(x))
        return p / np.sum(p)
        
    def forward(self, inputs, hprev):
            xs, hs, os, ycap = {}, {}, {}, {}
            hs[-1] = np.copy(hprev)
            for t in range(len(inputs)):
                xs[t] = np.zeros((self.vocab_size,1))
                xs[t][inputs[t]] = 1 # one hot encoding , 1-of-k
                whm1 = np.dot(self.W,hs[t-1])
                wxst = np.dot(self.U,xs[t]) 
                #hs[t] = np.tanh(np.dot(self.U,xs[t]) + np.dot(self.W,hs[t-1]) + self.b) # hidden state
                hs[t] = np.tanh(wxst + whm1 + self.b) # hidden state
                whyt = np.dot(self.V,hs[t])
                os[t] = whyt + self.c
                #os[t] = np.dot(self.V,hs[t]) + self.c # unnormalised log probs for next char
                ycap[t] = self.softmax(os[t]) # probs for next char
                if(self.log == True):
                    print("======== Step :" + str(t+1) + "============")
                    print("Wxh :\n", self.U)
                    print("xs[t] :\n", xs[t])
                    print("wxh *xt :\n", wxst)
                    print("Whh :\n", self.W)
                    print("hs[t-1] :\n", hs[t-1])
                    print("Whh*h(t-1) :\n",whm1)
                    print("b :\n", self.b)
                    print("hs[t] :\n", hs[t])
                    print("c :\n", self.c)
                    print("Why * hs[t] :\n", whyt)
                    print("os[t] :\n", os[t])
                    print("ycap[t] :\n", ycap[t])
                
            return xs, hs, ycap
        
    def backward(self, xs, hs, ps, targets):
            # backward pass: compute gradients going backwards
            dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
            db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
            dhnext = np.zeros_like(hs[0])
            for t in reversed(range(self.seq_length)):
                dy = np.copy(ps[t])
                #through softmax
                dy[targets[t]] -= 1 # backprop into y
                #calculate dV, dc
                dV += np.dot(dy, hs[t].T)
                dc += dc
                #dh includes gradient from two sides, next cell and current output
                dh = np.dot(self.V.T, dy) + dhnext # backprop into h
                # backprop through tanh non-linearity 
                dhrec = (1 - hs[t] * hs[t]) * dh  #dhrec is the term used in many equations
                db += dhrec
                #calculate dU and dW
                dU += np.dot(dhrec, xs[t].T)
                dW += np.dot(dhrec, hs[t-1].T)
                #pass the gradient from next cell to the next iteration.
                dhnext = np.dot(self.W.T, dhrec)
            # clip to mitigate exploding gradients
            for dparam in [dU, dW, dV, db, dc]:
                np.clip(dparam, -5, 5, out=dparam) 
            return dU, dW, dV, db, dc
    
    def loss(self, ps, targets):
            # calculate cross-entrpy loss
            return sum(-np.log(ps[t][targets[t],0]) for t in range(self.seq_length))
        
    
    def update_model(self, dU, dW, dV, db, dc):
        # parameter update with adagrad
        for param, dparam, mem in zip([self.U, self.W, self.V, self.b, self.c],
                                  [dU, dW, dV, db, dc],
                                  [self.mU, self.mW, self.mV, self.mb, self.mc]):
            mem += dparam*dparam
            param += -self.learning_rate*dparam/np.sqrt(mem+1e-8) # adagrad update
                
                
    def sample(self, h, seed_ix, n):
            x = np.zeros((self.vocab_size, 1))
            x[seed_ix] = 1
            ixes = []
            for t in range(n):
                h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
                y = np.dot(self.V, h) + self.c
                p = np.exp(y)/np.sum(np.exp(y))
                ix = np.random.choice(range(self.vocab_size), p = p.ravel())
                x = np.zeros((self.vocab_size,1))
                x[ix] = 1
                ixes.append(ix)
            return ixes

    def train(self, data_reader):
            iter_num = 0
            threshold = 0.01
            smooth_loss = -np.log(1.0/data_reader.vocab_size)*self.seq_length
            while (smooth_loss > threshold):
                if data_reader.just_started():
                    hprev = np.zeros((self.hidden_size,1))
                inputs, targets = data_reader.next_batch()
                xs, hs, ps = self.forward(inputs, hprev)
                dU, dW, dV, db, dc = self.backward(xs, hs, ps, targets)
                loss = self.loss(ps, targets)
                self.update_model(dU, dW, dV, db, dc)
                smooth_loss = smooth_loss*0.999 + loss*0.001
                hprev = hs[self.seq_length-1]
                if not iter_num%500:
                    sample_ix = self.sample(hprev, inputs[0], 4)
                    print( ''.join(data_reader.ix_to_char[ix] for ix in sample_ix))
                    print( "\n\niter :%d, loss:%f"%(iter_num, smooth_loss))
                iter_num += 1

    def train1(self, data_reader):
            iter_num = 0
            threshold = 0.01
            smooth_loss = -np.log(1.0/data_reader.vocab_size)*self.seq_length
            while (iter_num <1):
                if data_reader.just_started():
                    hprev = np.zeros((self.hidden_size,1))
                inputs, targets = data_reader.next_batch()
                xs, hs, ps = self.forward(inputs, hprev)
                dU, dW, dV, db, dc = self.backward(xs, hs, ps, targets)
                loss = self.loss(ps, targets)
                self.update_model(dU, dW, dV, db, dc)
                smooth_loss = smooth_loss*0.999 + loss*0.001
                hprev = hs[self.seq_length-1]
                if not iter_num%500:
                    sample_ix = self.sample(hprev, inputs[0], 4)
                    print( ''.join(data_reader.ix_to_char[ix] for ix in sample_ix))
                    print( "\n\niter :%d, loss:%f"%(iter_num, smooth_loss))
                iter_num += 1
    def predict(self, data_reader, start, n):

        #initialize input vector
        x = np.zeros((self.vocab_size, 1))
        chars = [ch for ch in start]
        ixes = []
        for i in range(len(chars)):
            ix = data_reader.char_to_ix[chars[i]]
            x[ix] = 1
            ixes.append(ix)

        h = np.zeros((self.hidden_size,1))
        # predict next n chars
        for t in range(n):
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            y = np.dot(self.V, h) + self.c
            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p = p.ravel())
            x = np.zeros((self.vocab_size,1))
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(data_reader.ix_to_char[i] for i in ixes)
        return txt

seq_length = 5
#read text from the "input.txt" file
data_reader = DataReader("input.txt", seq_length)        
rnn = RNN(hidden_size=4, vocab_size=data_reader.vocab_size,seq_length=seq_length,learning_rate=1e-1)
print(" U :", rnn.U)
print(" V :", rnn.V)
print(" W :", rnn.W)
print(" b :", rnn.b)
print(" c :", rnn.c)
rnn.train(data_reader)

retval = rnn.predict(data_reader, 'p', 5)
print("retval:",retval)

rnn1 = RNN(hidden_size=4, vocab_size=data_reader.vocab_size,seq_length=seq_length,learning_rate=1e-1)
rnn1.log = True
rnn1.U = rnn.U
rnn1.V = rnn.V
rnn1.W = rnn.W
rnn1.b = rnn.b
rnn1.c = rnn.c

print(" Final U :", rnn1.U)
print(" Final V :", rnn1.V)
print(" Final W :", rnn1.W)
print(" Final b :", rnn1.b)
print(" Finalc :", rnn1.c)
print("========= Calling with final Weights=======")
print("self.char_to_ix :",data_reader.char_to_ix)
rnn1.train1(data_reader)
#retval = rnn1.predict(data_reader, 'p', 5)
retval = rnn1.predict(data_reader, 'p', 5)
print("retval:",retval)