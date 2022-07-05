# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction
The fundamental purpose of neural style transfer (NST) is to combine two images - a content image and a style image - so that the output looks like the "content" image painted according to the "style" image. Arbitrary style transfer (AST) tries to synthesize a content image with the style of another image to create a third image that has never been seen before - hence it may be considered as the NST task where the model is an infinite-style model instead of a single-style one since we generate diverse outputs even for the same pair of inputs. 

In their paper ["Style-Aware Normalized Loss for Improving Arbitrary Style Transfer"](https://arxiv.org/pdf/2104.10064.pdf) Cheng et al. suggest an alternative loss function defined for the AST task that tries to solve some of the issues that the AST task's default loss function raises. It is published in one of the top conferences CVPR2021. 

The primary goal of this implementation is to train the well-known architectures SANet, LineerTransfer, ADAIn, and GoogleMagenta with a newly defined AST style loss function and demonstrate that this style-aware normalized loss function outperforms the default AST style loss function in every network. We also wish to replicate the authors' results to ensure that the style-aware normalized - or balanced - AST style loss functions behavior is more logical than the default AST style loss function.

## 1.1. Paper summary
The authors point out that current AST models face the problem of imbalanced style transferability (IST), in which the stylization intensity of the outputs varies greatly across different styles, and many stylized images suffer from under-stylization, in which only the dominant color is stylized, and over-stylization, in which the content is barely visible. To address this issue, they suggest a balanced AST style loss function that employs Gram-matrix like the default AST style loss function. On the other hand, the proposed method employs a normalizing term that is the theoretical upper bound for the classical AST layerwise style loss - or supremum - to weight estimated style losses. Authors test their proposal on 4 different networks - SANEt, LineerTransfer, ADAIn, and GoogleMagenta.

![](https://github.com/Neroxn/ceng501-final-project/blob/main/images/1.png?raw=true)

> Figure 1: The problem of under-stylization and over-stylization. In default style loss, the problem is visible as some stylized image content is not visible, or the style is only transferred with the color. It can be seen that the output of the proposed method is visually more plausible to the human eyes.
# 2. The method and my interpretation

## 2.1. The original method

Explain the original method.


## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Provide your names & email addresses and any other info with which people can contact you.
