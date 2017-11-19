""" GMM based clustering for x-y coordinates including visualization"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import mixture


def draw_2d_gaussians(model, colors, messages):
    fig, h = plt.subplots(1, 1)
    for mean, covarianceMatrix, color, message in zip(model.means_, model.covariances_, colors, messages):
        # get the eigen vectors and eigen values of the covariance matrix
        v, w = np.linalg.eigh(covarianceMatrix)
        v = 2.5 * np.sqrt(v)  # go to units of standard deviation instead of variance

        # calculate the ellipse angle and two axis length and draw it
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        currEllipse = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        currEllipse.set_alpha(0.5)
        h.add_artist(currEllipse)
        h.text(mean[0] + 7, mean[1] - 1, message, fontsize=13, color='blue')
    plt.ylim(-60, 440);
    plt.xlim(270, -270);
    plt.title('shot attempts')
    return fig


if __name__ == '__main__':
    data = pd.read_csv('../datasets/kobe.csv', index_col='shot_id')
    numGaussians = 13
    gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians, covariance_type='full',
                                                   init_params='kmeans', n_init=50,
                                                   verbose=0, random_state=5)
    gaussianMixtureModel.fit(data.loc[:, ['loc_x', 'loc_y']])

    # add the GMM cluster as a field in the dataset
    data['shotLocationCluster'] = gaussianMixtureModel.predict(data.loc[:, ['loc_x', 'loc_y']])
    # %% show gaussian mixture elipses of shot attempts
    plt.rcParams['figure.figsize'] = (13, 10)
    plt.rcParams['font.size'] = 15

    ellipseTextMessages = [str(100 * gaussianMixtureModel.weights_[x])[:4] + '%' for x in range(numGaussians)]
    ellipseColors = ['red', 'green', 'purple', 'cyan', 'magenta', 'yellow', 'blue', 'orange', 'silver', 'maroon',
                     'lime', 'olive', 'brown', 'darkblue']
    fig = draw_2d_gaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)

    fig.show()
    input()
