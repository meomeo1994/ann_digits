import sys
import numpy as np
import csv
import math
import os
from PIL import Image

img = Image.open('image_1.png')
arr = np.array(img)
    
class neutral_network:
    net = []
    phi = []
    out = []
    adj = []
    arr = []
    res = []
    n = 0
    
    def init(self,arr,n):
        self.arr = arr
        self.n = n
        for i in range(0,len(arr)):
            tmp = []
            for j in range(0,arr[i]):
                ll = []
                for l in range(0,n):
                    ll.append(0)
                tmp.append(ll)
            self.net.append(tmp)
            tmp = []
            for j in range(0,arr[i]):
                ll = []
                for l in range(0,n):
                    ll.append(0)
                tmp.append(ll)
            self.phi.append(tmp)
            tmp = []
            for j in range(0,arr[i]):
                ll = []
                for l in range(0,n):
                    ll.append(0)
                tmp.append(ll)
            self.out.append(tmp)
            
        for i in range(0,len(arr)-1):
            layer = []
            base = 0.5
            total = arr[i]*arr[i+1]
            for j in range(0,arr[i]):
                tmp = []
                for k in range(0,arr[i+1]):
                    tmp.append(base/total)
                    base += 1
                layer.append(tmp)
            self.adj.append(layer)
    
    def activat(self,x):
        if x > 40 :
            return 1.0
        if x < -40 :
            return 0.0
        return 1.0/(1.0 + math.pow(math.e,-x))

    def add(self,inp,res):
        self.res = res
        for i in range(0,len(inp)):
            for l in range(0,self.n):
                self.out[0][i][l] = inp[i][l]

    def putdata(self):
        print "putdata"
        fwriter = open("neutral.txt", "w")
        for i in self.adj:
            for j in i:
                for k in j:
                    fwriter.write(str(k)+'|')
        fwriter.close()

    def getdata(self):
        if not os.path.exists("neutral.txt"):
            return
        print "getdata"
        freader = open("neutral.txt", "r")
        s = freader.read()
        ar = s.split('|')
        dem = 0
        for i in range(0,len(self.arr)-1):
            for j in range(0,self.arr[i]):
                for k in range(0,self.arr[i+1]):
                    self.adj[i][j][k] = float(ar[dem])
                    dem += 1
        freader.close()
        
    def forward(self):
        last = len(self.arr)-1
        for i in range(0,len(self.arr)-1):
            for k in range(0,self.arr[i+1]):
                for l in range(0,self.n):
                    self.net[i+1][k][l] = 0
            for j in range(0,self.arr[i]):
                for k in range(0,self.arr[i+1]):
                    for l in range(0,self.n):
                        self.net[i+1][k][l] += self.out[i][j][l]*self.adj[i][j][k]
            for k in range(0,self.arr[i+1]):
                for l in range(0,self.n):
                    self.out[i+1][k][l] = self.activat(self.net[i+1][k][l]) 
        return self.out[last]
    
    def test(self,inp):
        last = len(self.arr)-1
        net = []
        out = []
        for i in range(0,len(self.arr)):
            tmp = []
            for j in range(0,self.arr[i]):
                tmp.append(0)
            net.append(tmp)
            tmp = []
            for j in range(0,self.arr[i]):
                tmp.append(0)
            out.append(tmp)
        for i in range(0,len(inp)):
            out[0][i] = inp[i]
        for i in range(0,len(self.arr)-1):
            for k in range(0,self.arr[i+1]):
                net[i+1][k] = 0
            for j in range(0,self.arr[i]):
                for k in range(0,self.arr[i+1]):
                    net[i+1][k] += out[i][j]*self.adj[i][j][k]
            for k in range(0,self.arr[i+1]):
                out[i+1][k] = self.activat(net[i+1][k])
        return out[last]

    def backward(self,alpha):
        last = len(self.arr)-1
        for i in range(0,self.arr[last]):
            for l in range(0,self.n):
                o = self.out[last][i][l]
                self.phi[last][i][l] = (o-self.res[i][l])*o*(1.0-o)

        for i in reversed(range(0,last)):
            for j in range(0,self.arr[i]):
                for l in range(0,self.n):
                    self.phi[i][j][l] = 0
                    for k in range(0,self.arr[i+1]):
                        self.phi[i][j][l] +=  self.phi[i+1][k][l]*self.adj[i][j][k]
                    o = self.out[i][j][l]
                    self.phi[i][j][l] *= o*(1.0-o)
                
            for j in range(0,self.arr[i]):
                for k in range(0,self.arr[i+1]):
                    for l in range(0,self.n):
                        self.adj[i][j][k] += -alpha*self.out[i][j][l]*self.phi[i+1][k][l]

def trainning():
    trainreader = csv.reader(open('train.csv', 'rb'), delimiter=' ', quotechar='|')

    inp = []
    res = []
    for i in range(0,10):
        res.append([])
    for i in range(0,784):
        inp.append([])
    first = True
    for i in trainreader:
        if first :
            first = False
            continue
        tmp = []
        for x in i[0].split(','):
            tmp.append(int(x))
        
        for j in range(0,10):
            if j == tmp[0] :
                res[j].append(1)
            else :
                res[j].append(0)
        for j in range(0,784):
            inp[j].append(tmp[j+1])

    ann = neutral_network()
    ann.init([784,10,10,10],42000)
    ann.add(inp,res)
    ann.getdata()
    for i in range(0,1000):
        ann.forward()
        ann.backward(math.pi)
        if i%10 == 0 :
            ann.putdata()
        

def testting():
    testreader = csv.reader(open('test.csv', 'rb'), delimiter=' ', quotechar='|')
    first = True
    for i in trainreader:
        if first :
            first = False
            continue
        tmp = []
        for x in i[0].split(','):
            tmp.append(int(x))
            
trainning()


















