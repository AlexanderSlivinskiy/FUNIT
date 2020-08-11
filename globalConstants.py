import torch

class GlobalConstants():
    outputPath = None
    precision = None
    usingApex = False

    inputchannels = None
    outputchannels = None


    def getPrecision():
        return GlobalConstants.checkIfSet(GlobalConstants.precision, "Precision", GlobalConstants.setPrecision.__name__)

    def setPrecision(precision):
        if (precision == "float16"):
            precision = torch.float16
        elif (precision == "float32"):
            precision = torch.float32
        elif (precision.upper() == "float16_APEX".upper()):
            precision = torch.float16
            GlobalConstants.usingApex = True
            print("Using APEX")
        elif (precision.upper() == "float32_APEX".upper()):
            precision = torch.float32
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
        return GlobalConstants.checkIfSet(GlobalConstants.outputPath, "Output path", GlobalConstants.setOutputPath.__name__)

    def setOutputPath(path):
        GlobalConstants.outputPath = path

    def setInputOutputChannels(inputCh, outputCh):
        GlobalConstants.inputchannels = inputCh
        GlobalConstants.outputchannels = outputCh

    def getInputChannels():
        return GlobalConstants.checkIfSet(GlobalConstants.inputchannels, "Input channels", GlobalConstants.setInputOutputChannels.__name__)

    def getOutputChannels():
        return GlobalConstants.checkIfSet(GlobalConstants.outputchannels, "Output channels", GlobalConstants.setInputOutputChannels.__name__)


    def checkIfSet(x, var_name, func_name):
        if (x is None):
            raise Exception(""+var_name+" is not set in GlobalConstants. Use GlobalConstants."+func_name+" first.")
        else:
            return x

    