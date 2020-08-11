from skimage.exposure import rescale_intensity
import torch
import numpy as np
from imgaug import augmenters as iaa
from globalConstants import GlobalConstants


def transformTo3Tuple(image):
    if len(image.shape) > 3:
        image = np.max(image, axis=0)
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=0)

    #if image.shape[0] < image.shape[-1]:
    #    image = np.transpose(image, (1, 2, 0))
    #if image.shape[-1] == 1:
    #    image = np.repeat(image, 3, axis=-1)

class RescaleToZeroOne(object):
    #Converts a PyTorch Tensor with RGB Range [0, 255] to PyTorch Tensor [0,1]

    def __call__(self, pic):
        """
        Args:
            pic (tensor or numpy.ndarray): Image to be rescaled.
        Returns:
            Tensor: Converted image.
        """
        #return pic/255
        return pic/pic.max()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RescaleToOneOne(object):
    #Converts a PyTorch Tensor with RGB Range [0, 255] to PyTorch Tensor [0,1]

    def __call__(self, pic):
        """
        Args:
            pic (tensor or numpy.ndarray): Image to be rescaled.
        Returns:
            Tensor: Converted image.
        """
        #half does not support the max operation on the CPU but only on CUDA
        if (pic.dtype == torch.float16):
            maximum = pic.numpy().max()
            pic = pic.numpy()
            maximum = pic.max()
            res = ((pic / maximum) *2)-1
            res = torch.from_numpy(res)
            res = res.half()
        else:
            maximum = pic.max()
            res = ((pic/ maximum) *2)-1
        return res

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def reverse(tensor):
        (tensor.cpu().detach()[0]+1)/2*255

class ToTensor(object):
    
    def __call__(self, pic):
        """
        if (pic.dtype == 'uint16'):
            print("anything else than double!!")
            if (pic.max()<32768):
                pic = pic.astype('int16')
            else:
                pic = pic.astype('int32')
        """
        if (GlobalConstants.getInputChannels()==3 and pic.shape[2]==3):
            pic=pic.transpose()
        elif (GlobalConstants.getInputChannels()==1 and len(pic.shape)==2):
            pic=pic.reshape((1, pic.shape[0], pic.shape[1]))

        tensor = torch.from_numpy(pic.copy())
        if (not GlobalConstants.usingApex):
            tensor = GlobalConstants.setTensorToPrecision(tensor)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DynamicResize(iaa.Resize):
    def __init__(self, desired_size):
        self.desired_size = desired_size

    # This is willingly NOT exponentially growing, since this could result and cropping half the picture
    def get_closest_factor(self, current_size):
        x = 1
        while (self.desired_size <= current_size//(x+1)):
            x+=1
        return x


    def __call__(self, pic):
        #Assume that we have (3,y,x)-shape
        print(pic.shape)
        new_x = self.get_closest_factor(pic.shape[2])
        new_y = self.get_closest_factor(pic.shape[1])
        scalar = min(new_x, new_y) #We want to keep proportions
        resizer = iaa.Resize({"height":pic.shape[1]//scalar, "width":pic.shape[2]//scalar})
        return resizer(pic)


    def __repr__(self):
        return self.__class__.__name__ + '()'

class PrintInputShape(object):
    #Converts a PyTorch Tensor with RGB Range [0, 255] to PyTorch Tensor [0,1]

    def __call__(self, pic):
        print(pic.shape)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'