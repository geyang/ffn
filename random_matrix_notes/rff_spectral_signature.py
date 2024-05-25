#
# See the exercise "RandomMatrixTheory.pdf" from RandomMatrixTheory.html
# in  http://www.physics.cornell.edu/sethna/StatMech/ComputerExercises/
#
"""Random Matrix Theory exercise"""

# Import numeric and plotting routines
# import blahDeBlah as blee allows abbreviations
import numpy

ra = numpy.random
la = numpy.linalg
import pylab


def GOE(N):
    """
    Creates an NxN element of the Gaussian Orthogonal Ensemble,
    by creating an NxN matrix of Gaussian random variables
      using the random array function ra.standard_normal([shape])
      with [shape] = (N,N).
    and adding it to its transpose (applying numpy.transpose to the matrix)
    #
    Typing GOE(4) a few times, check it's symmetric and random
    """
    m = ra.standard_normal((N, N))
    m = m + numpy.transpose(m)
    return m


def GOE_Ensemble(num, N):
    """
    Creates a list "ensemble" of length num of NxN GOE matrices.
    Starts with ensemble equalling an empty list []
    Then, for n in range(num) uses the function ensemble.append
    to add GOE(N) to the list.
    #
    Check GOE_Ensemble(3,2) gives 3 2x2 symmetric matrices
    """
    # ensemble = [GOE(N) for n in range(num)]
    ensemble = []
    for n in range(num):
        ensemble.append(GOE(N))
    return ensemble


def CenterEigenvalueDifferences(ensemble):
    """
    For each symmetric matrix in the ensemble, calculates the difference
    between the two center eigenvalues of the ensemble, and returns the
    differences as an array, after chopping off any tiny imaginary parts.
    #
    Given an ensemble of symmetric matrices,
    finds their size N
        len(m) gives the number of rows in m
    starts with an empty list of differences
    for each m in ensemble
      finds the eigenvalues
        la.eigvals(m) gives the eigenvalues of m
      sorts them
        numpy.sort(e) sorts a list from smallest to largest
      appends eigenvalue[N/2] - eigenvalue[N/2-1] to the list
         (Note: python lists go from 0 ... N-1, so for N=2 this
          gives the difference between the only two eigenvalues)
    Converts the list of differences diff into an array.
      Lists are python objects: they can be appended to but not multiplied.
      Arrays are numpy objects that look the same as lists, but
      can't be appended and can be multiplied.
      Use d = numpy.array(d) to convert from a list to an array.
    Takes the real part of the difference array (rounding errors)
      For a numpy d, d.real is the real part
    returns the (real part of the) array of differences
    #
    Check
       ensemble = GOE_Ensemble(3,2)
       CenterEigenvalueDifferences(ensemble)
    gives three positive numbers, that look like eigenvalues of the 2x2 matrices
    """
    # Size of matrix
    N = len(ensemble[0])
    diffs = []
    for mat in ensemble:
        eigenvalues = numpy.sort(la.eigvals(mat))
        diffs.append(eigenvalues[N // 2] - eigenvalues[N // 2 - 1])
    diffs = numpy.array(diffs)
    diffs = diffs.real
    return diffs


def CenterDiffHistogram(ensemble, bins=30, showPlot=True):
    """
    Calculates the center eigenvalue difference of an ensemble of
    NxN matrices, using GOE_ensemble and CenterEigenvalueDifferences.
    Finds the averages diffAve of the differences.
      numpy.sum(d) gives the total; len(d) gives the num in the ensemble
    Calculates an array of normalized differences, dividing diff by diffAve
    Plots a histogram of the normalized differences
      pylab.hist(d, bins=bins, normed=1) prepares a plot of a histogram with
      total area one (normalized to a probability distribution).
        The notation def f(x, y=3) in a function definition gives a default
	value of 3 for the optional input variable y. The same notation
	in calling the function f(x, y=2) will overload the default value
	of y with 2. Thus hist(d, bins=bins) overloads the default
	value of bins in the histogram with the input variable "bins".
      pylab.show() should be run (to display the graph) if showPlot==True.
      Warning: pylab currently freezes the display, so you'll need to
      close the graph before typing more Python commands.
    #
    Check that
       ensemble = GOE_Ensemble(3,2)
       CenterDiffHistogram(ensemble)
    gives spikes at the three eigenvalue differences you found with
       CenterEigenvalueDifferences(ensemble)
    Then run with ensembles of thousands of matrices, with size N from 2-10
    """
    diffs = CenterEigenvalueDifferences(ensemble)
    meanDiff = numpy.sum(diffs) / len(diffs)
    diffsNormalized = diffs / meanDiff
    import matplotlib.pyplot as plt
    plt.hist(diffsNormalized, bins=bins, density=1)
    if showPlot == True:
        pylab.show()


def Wigner(s):
    """
    Returns the Wigner surmise for the probability distribution rho(s)
    for the eigenvalue differences in the GOE ensemble,
    rho(s) = (pi s/2) exp(-pi s^2/4)
    Python provides a number of complications:
      s^2 is s**2
      In Python (and in some other languages), always divide by floats
      unless you want the value truncated (s/2.0 not s/2)
      pi and exp are not Python functions, but numpy functions
    The Wigner surmise is correct only for 2x2 GOE matrices, but it's
    easily derived and is quite close to the true (much messier) answer
    for larger matrices.
    #
    Test that Wigner(1.0) is reasonable, that Wigner(0.0) is zero,
    and that Wigner(10.0) is small.
    """
    return (numpy.pi * s / 2.0) * numpy.exp(-numpy.pi * s ** 2 / 4.)


def CompareEnsembleWigner(ensemble, bins=30):
    """
    Plots the center eigenvalue difference histogram using CenterDiffHistogram
    with showPlot=False.
    Then creates an array of s values using numpy.arange from 0.0 to 4.0
    with a step of 0.01.
    If the function Wigner is defined using only multiplication, powers,
    and simple numpy functions like exponentials, Wigner(sArray) should
    return an array of the Wigner distribution evaluated at the values of s.
    pylab.plot(sArray,theory) should provide the theory curve;
    pylab.show() then displays both histogram and Wigner surmise.
    #
    Test the agreement with thousands of matrices and N=2.
    Test that the agreement remains good with N from 4-10, and at least 30 bins.
    """
    CenterDiffHistogram(ensemble, bins=bins, showPlot=False)
    sArray = numpy.arange(0.0, 4.0, 0.01)
    theory = Wigner(sArray)
    pylab.plot(sArray, theory)
    pylab.show()


def PM1(N):
    """
    Creates a symmetric NxN matrix with random entries of +-1.
    I generated an asymmetric array ra.randint, and then set the
    bottom left equal to the top right in a double loop.
      #
      ra.randint(min, max+1, (N,N)) will generate random integers
      in the range (min, min+1, ..., max) inclusive.
        (Notice that the range starts at min but ends one BEFORE max:
	this makes sense in python, where length-N arrays start at zero
	and end at N-1.)
      You'll want to generate a random matrix of zeros and ones,
      and then multiply by two and subtract one.
      #
      Symmetrizing needs i in range(N) and j in range(i)
      [range, like arrays and randint, stops one before the end]
      #
    Test first, by generating some matrices and making sure they're symmetric
    and +-1.
    """
    m = ra.randint(0, 2, (N, N)) * 2.0 - 1.0
    for i in range(N):
        for j in range(i):
            m[j, i] = m[i, j]
    return m


def PM1_Ensemble(num, N):
    """
    Generates a +-1 ensemble, as for GOE_Ensemble above.
    Use with CompareEnsembleWigner to test that N=2 looks different
    from the Wigner ensemble, but larger N looks close to the GOE ensemble.
    #
    This is a powerful truth: the eigenvalue differences of very general classes
    of symmetric NxN matrices all have the same probability distribution
    as N goes to infinity. This is what we call universality.
    """
    # ensemble = [PM1(N) for n in range(num)]
    ensemble = []
    for n in range(num):
        ensemble.append(PM1(N))
    return ensemble


def yesno():
    response = input('    Continue? (y/n) ')
    if len(response) == 0:  # [CR] returns true
        return True
    elif response[0] == 'n' or response[0] == 'N':
        return False
    else:  # Default
        return True


def demo():
    """Demonstrates solution for exercise: example of usage"""
    print("Random Matrix Theory Demo")
    print("  GOE 2x2 vs. Wigner")
    CompareEnsembleWigner(GOE_Ensemble(1000, 2))
    if not yesno(): return
    print("  +-1 2x2 vs. Wigner")
    CompareEnsembleWigner(PM1_Ensemble(1000, 2))
    if not yesno(): return
    print("  +-1 10x10 vs. Wigner")
    CompareEnsembleWigner(PM1_Ensemble(1000, 10))


if __name__ == "__main__":
    demo()
