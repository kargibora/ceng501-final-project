# Style-Aware Normalized Loss for Improving Arbitrary Style Transfer

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

Many different AST losses has been proposed but the most employed one is the original NST loss which is defined as :
![](https://latex2png.com/pngs/e8123b2bf7febeccff494153e97fdf8f.png)
> Figure 2.1: Original NST loss.

Original NST loss composed of 
- Content loss between the content image *C* and the product image *P*
- Style loss between the style image *S* and the product image *P*
- Weighting term  β for weighting the total loss.
Explain the original method.

To encode the information, many models use ImageNet pretrained VGG network for extracting features from content image *C*, style image *S* and product image *P*. Content loss is usually calculated by comparing the extracted features of the *P* and *C* whereas style loss is calculated by comparing the Gram matrices of the extracted features of *P* and *S*.  In practice, content and style features are calculated using different parts of the VGG network so in general, we can define default NST losses as :


<p align="center">
  <img src="https://github.com/Neroxn/ceng501-final-project/blob/main/images/default_losses.png?raw=true"/>
</p>

> Figure 2.2: Default NST loss. Subscript shows whether the loss is **c**ontent loss or **s**tyle loss and superscript denotes which layer of VGG network is used to extract the feature vectors of the images. $G$ is a function that returns the Gram matrix of the input and **MSE** is the mean-squared error. We can use $w^l$ to give a weight for specific loss values however they are usually set up to 1.

Authors identify the problem of using such style loss function by explaining that style losses for different style images can differ by more than 1.000 times for both randomly initialized and fully trained AST model. So style with small loss ranges leads to under-stylization and style with large loss ranges leads to oveer-stylization. This core problem also leads up to a interseting negative outcome for using unbalanced AST style loss function.

<p align="center">
  <img src="https://user-images.githubusercontent.com/44094497/177416747-4c0af3fc-a690-49ec-97e4-91f041dfbb3f.png"/>
</p>

> Figure 2.3: Distribution of classic Gram matrix-based stlye losses for four AST methods. Notice that understylized images attain lowest losses whereas overstylized images attains higher style loss than others. Inituatively, this seems wrong.

Authors first define classic AST style loss as :

<p align="center">
  <img src="https://user-images.githubusercontent.com/44094497/177417834-24caf925-98ba-4669-83a1-3c2718183681.png" />
</p>


> Figure 2.4: Definition of the classic AST style loss. It's only difference with the loss definde in the Figure 2.2 is the used layers for extracting the features of images.
 
For extracting the features of images, they have used $F_{b1}^{r2}$, $F_{b2}^{r2}$, $F_{b3}^{r3}$ and $F_{b4}^{r4}$ where $F_{bi}^{rj}$ denotes j-th *ReLU* layer of the i-th convolutional block of VGG-16. Authors tries to balance this default loss function by using a task-dependent normalization term $V^{l}(S,P)$. Figure 2.5 shows the proposed style loss function.



![image](https://latex2png.com/pngs/d7a8f7201cc204d9a17a00c0c7c0ee30.png)

> Figure 2.5: Definition of the balanced AST style loss. Notice that we have a normalizing term which tries to balance the style loss with respect to the style image it is given.

Only problem left is finding such $V^{l}(S,P)$. Authors claims that theoritical upper-bounds for the classic AST layerwise style losses is a great candidate and propose their balanced AST style loss as:


<p align="center">
  <img src="https://github.com/Neroxn/ceng501-final-project/blob/main/images/balanced_loss.png?raw=true" />
</p>

> Figure 2.6: Proposed balanced AST style loss.


## 2.2. My interpretation 

In the paper, some part of the algorithms are left unclear. For example the reason why authors suggested the extraction functions as $F_{b1}^{r2}$, $F_{b2}^{r2}$, $F_{b3}^{r3}$ and $F_{b4}^{r4}$ is not well-understood as no analysis or a logical statement is proposed for choosing such layers for encoding the image into feature vectors. 

Secondly, the term $N^l$ in the Figure 2.6 is defined as a "that is equal to the product of spatial dimensions of the feature tensor at layer $l$." Spatial dimensions of the feature tensor is unclear and we interpreted this term as $C^2$ where $C$ denotes the channel size of the input at layer $l$ since height and width of the gram matrix of the input at layer $l$ is equal to the channel size $C$. 

Lastly as it can be seen in the Figure 2.3, the classic style losses are nearly $10^9$. Such high style loss can be problematic for many problems so in practice and other litearatures, this term is usually normalized by dividing Gram matrix that is produced for an input by it's dimension. For this reason, instead of using un-normalized version of default AST style loss, we have normalized it for better stability. However since some of the analyzes requires the un-normalized versions, such analyzes is done with the test dataset.


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
