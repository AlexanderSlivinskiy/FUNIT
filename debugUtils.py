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

    def printCheckpoint(self, function=None, content=""):
        if function == None:
            f = self.funcName
        else:
            f = function.__name__
        print("==="+self.prefix+" : "+self.className+"."+f+", checkpoint:",self.index,"===")
        if (content != ""):
            print(content)
        self.index+=1

    def checkForNaNandInf(self, tensor, msg=""):
        nan = torch.isnan(tensor)
        inf = torch.isinf(tensor)
        if (torch.sum(nan) != 0):
            print("INPUT IS NAN IN FORWARD PASS!!", torch.sum(nan))
            self.printCheckpoint(content=msg)
        if (torch.sum(inf) != 0):
            print("INPUT IS INF IN FORWARD PASS!!", torch.sum(inf))
            self.printCheckpoint(content=msg)
            #print(tensor)

    def printgradnorm(self, cls, grad_input, grad_output):
        print('Inside ' + cls.__class__.__name__ + ' backward')
        #print('')
        #print('grad_input: ', type(grad_input))
        #print('grad_input[0]: ', type(grad_input[0]))
        #print('grad_output: ', type(grad_output))
        #print('grad_output[0]: ', type(grad_output[0]))
        #print('')
        #print('grad_input size:', grad_input[0].size())
        #print('grad_output size:', grad_output[0].size())
        #print('grad_input norm:', grad_input[0].norm())
        print('grad_output_max:', grad_output[0].max())
        #print(grad_output)