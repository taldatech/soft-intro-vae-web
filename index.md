## Welcome to GitHub Pages


<h1 align="center">
  <br>
Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders
  <br>
</h1>
  <p align="center">
    <a href="https://github.com/taldatech">Tal Daniel</a> •
    <a href="https://avivt.github.io/avivt/">Aviv Tamar</a>

  </p>

  [Code]() [Paper]()

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/soft-intro-vae-pytorch"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>


<p align="center">
  <img src="./assets/ffhq_samples.png">
</p>

# Soft-IntroVAE

> **Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders**<br>
> Tal Daniel, Aviv Tamar<br>
>
> **Abstract:** *The recently introduced introspective variational autoencoder (IntroVAE) exhibits outstanding image generations, and allows for amortized inference using an image encoder. The main idea in IntroVAE is to train a VAE adversarially, using the VAE encoder to discriminate between generated and real data samples. However, the original IntroVAE loss function relied on a particular hinge-loss formulation that is very hard to stabilize in practice, and its theoretical convergence analysis ignored important terms in the loss.
In this work, we take a step towards better understanding of the IntroVAE model, its practical implementation, and its applications. We propose the Soft-IntroVAE, a modified IntroVAE that replaces the hinge-loss terms with a smooth exponential loss on generated samples. This change significantly improves training stability, and also enables theoretical analysis of the complete algorithm. Interestingly, we show that the IntroVAE converges to a distribution that minimizes a sum of KL distance from the data distribution and an entropy term. We discuss the implications of this result, and demonstrate that it induces competitive image generation and reconstruction. Finally, we describe an application of Soft-IntroVAE to unsupervised image translation, and demonstrate compelling results.*



You can use the [editor on GitHub](https://github.com/taldatech/soft-intro-vae-web/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.
$$ X^2 = C $$

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/taldatech/soft-intro-vae-web/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
