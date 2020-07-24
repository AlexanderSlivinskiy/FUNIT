class GlobalConstants():
    outputPath = None

    def getOutputPath():
        if (GlobalConstants.outputPath is None):
            raise Exception("output path is not set. Use setOutputPath first")
        else:
            return GlobalConstants.outputPath

    def setOutputPath(path):
        GlobalConstants.outputPath = path