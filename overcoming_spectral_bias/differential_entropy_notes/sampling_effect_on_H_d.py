from cmx import doc
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    doc @ """
    # Effect of Sampling On Differential Entropy
    
    The goal: vary the number of samples from the distribution, and compare
    the sample-based estimate against those computed from the analytical solution.
    This is done on the scipy documentation page [^1].
    
    [^1]: [scipy.org ... /scipy.stats.differential_entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.differential_entropy.html)
    
    ## A Simple Example
    
    Consider a uniform distribution between $[0, 1/2]$
    $$
    p(x) = \\begin{cases}
        2 & \\text{if } x \in [0, 1/2] \\\\
        0 & \\text{otherwise. }
    \end{cases}
    $$
    The differential entropy is $-\log 2$. 
    """
    with doc:
        def H_d(ps):
            ps_norm = ps / ps.sum()
            return - np.sum(np.log(ps) * ps_norm)

        def get_H_d(N):
            xs = np.linspace(0, 1 / 2, N)
            ps = np.ones(N) * 2
            h_d = H_d(ps)
            doc.print(f"N={N} => h_d={h_d:0.2f}")

        for n in [101, 201, 401, 801, 1601]:
            get_H_d(n)

    doc @ """
    The differential entropy does not depend on the number of points in the 
    discretized distribution as long as the discretization is done properly.
    """

    doc.flush()