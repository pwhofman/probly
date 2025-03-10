import matplotlib.pyplot as plt
import mpltern

def simplex_plot(probs):
    """
    Plot probability distributions on the simplex.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='ternary')
    ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2])
    return fig, ax
