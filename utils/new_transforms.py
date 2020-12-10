"""
Downloaded from https://github.com/sedab/PathCNN
All cretids go to the PathCNN developers.
"""

import numpy as np
import math
import random
from PIL import Image, ImageOps, ImageEnhance
import collections
import types
import cv2 #ADD
import staintools #ADD
import histomicstk as htk #ADD

"""
Taken directly from https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
Latest update that is not currently deployed to pip.

All credits to the torchvision developers.
"""

accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
    
def crop(img, i, j, h, w):
    """Crop the given PIL.Image.
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL.Image to the given size.
    Args:
        img (PIL.Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL.Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
    
def vflip(img):
    """Vertically flip the given PIL.Image.
    Args:
        img (PIL.Image): Image to be flipped.
    Returns:
        PIL.Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

#ADD
class StainAugStainTools(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            self.to_augment = staintools.LuminosityStandardizer.standardize(np.array(img).astype('uint8'))
            self.augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
            self.augmentor.fit(self.to_augment)
            augmented_img = self.augmentor.pop()
            return Image.fromarray(augmented_img.astype('uint8')) # .astype(float64)
        else:
            return img

#ADD
class ShapeAugCanny(object):
    def __init__(self, prob, low, high):
        self.prob = prob
        self.low = low
        self.high = high
    def __call__(self, img):
        if random.random() < self.prob:
            img = np.array(img).astype('uint8')
            canny = cv2.Canny(img[:,:,2],self.low,self.high)
            img[:,:,0]=canny
            img[:,:,1]=canny
            img[:,:,2]=canny
            return Image.fromarray(img.astype('uint8')) # .astype(float64)
        else:
            return img

#ADD
class ShapeAugCanny2(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = np.array(img).astype('uint8')
            canny = cv2.Canny(img[:,:,2],100,200)
            img[:,:,0]=canny
            return Image.fromarray(img.astype('uint8')) # .astype(float64)
        else:
            return img

#ADD ref = '/home/rikiya/projects/hcc/data/ref_hcc.png'
class ShapeAugHistTK(object):
    def __init__(self, path2ref, prob):
      self.path2ref = path2ref
      self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            target = staintools.read_image(self.path2ref)
            to_transform = staintools.LuminosityStandardizer.standardize(np.array(img).astype('uint8'))
            target = staintools.LuminosityStandardizer.standardize(target)
            to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
            normalizer = staintools.StainNormalizer(method='vahadane')
            normalizer.fit(target)
            im_nmzd = normalizer.transform(to_transform)
            stainColorMap = {
                'hematoxylin': [0.65, 0.70, 0.29],
                'eosin':       [0.07, 0.99, 0.11],
                'dab':         [0.27, 0.57, 0.78],
                'null':        [0.0, 0.0, 0.0]
                }
            stain_1 = 'hematoxylin'   # nuclei stain
            stain_2 = 'eosin'         # cytoplasm stain
            stain_3 = 'null'          # set to null of input contains only two stains
            W = np.array([stainColorMap[stain_1],
                        stainColorMap[stain_2],
                        stainColorMap[stain_3]]).T
            im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains
            a = htk.filters.edge.gaussian_grad(im_stains[:,:,0], sigma=0.16)
            b = ((a.dx + (a.dx.max()- a.dx.min()))/(a.dx.max()- a.dx.min())*255.).astype('uint8')
            im_stains[:,:,2] = b.astype('uint8')
            c = Image.fromarray(im_stains.astype('uint8'))

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # b_clahe = clahe.apply(b)
            # Image.fromarray(b_clahe.astype('uint8'))
            # c = ((b_clahe>180)*255).astype('uint8')
            # c = Image.fromarray(c.astype('uint8'))
            return c
        else:
            return img

class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return resize(img, self.size, self.interpolation)

class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return vflip(img)
        return img

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL.Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL.Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL.Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL.Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL.Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL.Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL.Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL.Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
    See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        img (PIL.Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img

def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL.Image and resize it to desired size.
    Notably used in RandomResizedCrop.
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL.Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img

class RandomResizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation

    @staticmethod
    def get_params(img):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)
    
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image.
        Returns:
            PIL.Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

def adjust_rotation(img, degree=90):
    """Roatete  the given PIL.Image.
    Args:
        img (PIL.Image): Image to be flipped.
        degree: Angle to rotate: 0 to 360
    Returns:
        PIL.Image:  Rotated image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        
    if degree<0: 
        raise ValueError('Negative rotation - Select degree between 0 and 360')
        
    if degree>360: 
        raise ValueError('Negative rotation - Select degree between 0 and 360')

    return img.rotate(degree)

class ColorJitterRotate(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
        rotation: Rotate image randomly 0 to the defined parameter, fixed between (0, 90, 180, 270)
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, rotation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.rotation = rotation

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, rotation):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))
        
        if rotation > 0: 
            rotation_factor = np.random.uniform(0, rotation)
            rotation_factor =  min([0,90,180,270,360], key=lambda x:abs(x-rotation_factor))
            transforms.append(Lambda(lambda img: adjust_rotation(img, rotation_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image.
        Returns:
            PIL.Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue, self.rotation)
        return transform(img)

class RandomRotate(object):
    """Randomly change the Rotation of an image (0/90/180/270).
    
    Args:
        rotation: Rotate image randomly to the defined parameter, fixed between (0, 90, 180, 270)
        
    """
    def __init__(self, rotation=0):
        self.rotation = rotation

    @staticmethod
    def get_params(rotation):
        
        transforms = []
        rotation_factor = np.random.randint(0, 4, 1) * 90
        transforms.append(Lambda(lambda img: adjust_rotation(img, rotation_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image.
        Returns:
            PIL.Image: Randomly rotated image.
        """
        transform = self.get_params(self.rotation)
        
        return transform(img)
