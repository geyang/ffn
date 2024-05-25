from cmx import doc
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    doc @ """
    # Effect of Change of Variable On Differential Entropy
    
    The goal: Investigate the effect of the change of variable, by varying the 
    scale of the distribution and looking at the entropy. We can look at both
    the differential entropy analytically, and the sample-based estimates [^1].
    
    [^1]: [scipy.org ... /scipy.stats.differential_entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html)
    
    ## A Simple Example
    
    Consider a uniform distribution between $[a, b]$
    $$
    p(x) = \\begin{cases}
        \\frac 1 {\\vert b - a \\vert} & \\text{if } x \in [a, b] \\\\
        0 & \\text{otherwise. }
    \end{cases}
    $$
    The differential entropy is $\log \left(\\vert b - a \\vert\\right)$. When 
    `a=0` and `b=0.5`, $H(x) = - \log 2$. 
    
    We can verify this numerically. First let's define the entropy function
    """
    with doc:
        def H_d(ps):
            ps_norm = ps / ps.sum()
            return - np.sum(np.log(ps) * ps_norm)

    doc @ """
    We can plot the delta against $\log(b - a)$ -- it turned out to be zero
    across the range variants.
    """
    with doc:
        def get_H_d(a, b, N=201):
            xs = np.linspace(a, b, N)
            ps = np.ones(N) / (b - a)
            h_d = H_d(ps)
            delta = h_d - np.log(b - a)
            doc.print(f"a={a}, b={b} => h_d={h_d:0.2f}, δ={delta}")

        for b in [1/4, 1/2, 1, 2, 4]:
            get_H_d(0, b)

    doc.flush()