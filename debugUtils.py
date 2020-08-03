import torch

def printCheckpoint(index, funcName, className="", prefix=""):
    print("==="+prefix+" : "+className+"."+funcName+", checkpoint:",index,"===")

class Debugger():
    #Expected call: debug = Debugger(self.function, self, prefix)
    def __init__(self, func=None, classSelf=None, prefix=""):
        if (func != None):
            self.funcName = func.__name__
        else:
            self.funcName = ""
        if (classSelf != None):
            self.className = classSelf.__class__.__name__
        else:
            self.className = ""
        self.prefix = prefix
        self.index = 0

    def printCheckpoint(self, content=""):
        print("==="+self.prefix+" : "+self.className+"."+self.funcName+", checkpoint:",self.index,"===")
        if (content != ""):
            print(content)
        self.index+=1

    def checkForNaNandInf(self, tensor):
        nan = torch.isnan(tensor)
        inf = torch.isinf(tensor)
        if (torch.sum(nan) != 0):
            print("INPUT IS NAN IN FORWARD PASS!!", torch.sum(nan))
            self.printCheckpoint()
        if (torch.sum(inf) != 0):
            print("INPUT IS INF IN FORWARD PASS!!", torch.sum(inf))
            self.printCheckpoint()