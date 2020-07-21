def printCheckpoint(index, funcName, className="", prefix=""):
    print("==="+prefix+" : "+className+"."+funcName+", checkpoint:",index,"===")

class Debugger():
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