import torch

class GlobalConstants():
    outputPath = None
    precision = None
    usingApex = False

    def getPrecision():
        if (GlobalConstants.precision is None):
            raise Exception("precision is not set. Use setPrecision first")
        else:
            return GlobalConstants.precision

    def setPrecision(precision):
        if (precision == "float16"):
            precision = torch.float16
        elif (precision == "float32"):
            precision = torch.float32
        elif (precision.upper() == "float16_APEX".upper()):
            precision = torch.float16
            GlobalConstants.usingApex = True
            print("Using APEX")
        print("Set precision to:", precision)
        GlobalConstants.precision = precision

    def setTensorToPrecision(tensor):
        precision = GlobalConstants.getPrecision()
        if (precision == torch.float16):
            return tensor.half()
        elif (precision == torch.float32):
            return tensor.float()

    def getOutputPath():
        if (GlobalConstants.outputPath is None):
            raise Exception("output path is not set. Use setOutputPath first")
        else:
            return GlobalConstants.outputPath

    def setOutputPath(path):
        GlobalConstants.outputPath = path

    