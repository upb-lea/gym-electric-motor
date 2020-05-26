---
title: 'Gym-electric-motor (GEM): A Python library for the simulation of electric motors'
tags:
  - Python
  - electric drive control
  - electric motors
  - OpenAI Gym
  - power electronics
  - reinforcement learning
authors:
  - name: Praneeth Balakrishna
    affiliation: 1
    
  - name: Gerrit Book
    affiliation: 1
    
  - name: Wilhelm Kirchgässner
    orcid: 0000-0000-0000-0000
    affiliation: 1
    
  - name: Maximilian Schenke
    orcid: 0000-0000-0000-0000
    affiliation: 1
    
  - name: Arne Traue
    affiliation: 1
    
  - name: Oliver Wallscheid
    orcid: 0000-0000-0000-0000
    affiliation: 1
    
affiliations:
 - name: Department of Power Electronics and Electrical Drives, Paderborn University, Germany
   index: 1
 - name: Institution 2
   index: 2
date: 26. May 2020
bibliography: Literature.bib
---

# Summary

The ``gym-electric-motor`` (``GEM``) library provides simulation environments for 
electrical drive systems, and therefore allows to easily design and analyze drive control
solutions in Python. Since ``GEM`` is strongly inspired by OpenAI's ``gym``, it 
is particulary well-equipped for (but not limited to) applications in the field of 
reinforcement-learning-based control algorithms. The API allows to overwrite
the default simulation parameters to e.g. parametrize a specific motor or configure
the desired load behavior.

# Field of Application

Electric drive systems are an important topic both in academic and 
industrial research due to the worldwide availability and deployment of such 
plants. Control algorithms for these systems are usually designed, parametrized and 
tested within ``MATLAB - Simulink`` [@], which is developed and marketed specifically for
such engineering tasks. In the more recent past, however, commercial software like
``MATLAB`` has difficulties to stay on par with the functionality and flexibility offered 
by open-source libraries that are available for Python. Moreover, the latest efforts
concerning industrial application of reinforcement-learning control algorithms heavily 
depend on Python packages like ``Keras`` [@] or ``Tensorflow`` [@]. To allow easy 
access to a drive simulation environment, the ``GEM`` library has been developed.
  



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References