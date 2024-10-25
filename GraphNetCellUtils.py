import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


'''
Some visualization functions
'''

# determine the baseline
def getBaseline(data_list_test):
    test_y_values = []
    test_SI_values = []
    # Iterate over the dataset and collect the feature values
    for data in data_list_test:
        test_y_values.append(data.y.item())
        SI = np.median(data.perimeter/np.sqrt(data.area))
        test_SI_values.append(SI)

    # linear fitting
    test_SI_values = np.array(test_SI_values)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_SI_values, test_y_values)
    predicted_y_values = slope * test_SI_values + intercept
    correlation = r_value
    mean_squared_error = np.mean((predicted_y_values - test_y_values) ** 2)

    return correlation, mean_squared_error


# scatter plot mobility against cell density and median shape index
def visualizeDataDistribution(data_list_train, data_list_test):
    train_y_values = []
    train_n0_values = []
    train_SI_values = []
    test_y_values = []
    test_n0_values = []
    test_SI_values = []
    # Iterate over the dataset and collect the feature values
    for data in data_list_train:
        train_y_values.append(data.y.item())
        n0 = torch.tensor(np.array(len(data.x)), dtype=torch.float)
        train_n0_values.append(n0)
        SI = np.median(data.perimeter/np.sqrt(data.area))
        train_SI_values.append(SI)

    for data in data_list_test:
        test_y_values.append(data.y.item())
        n0 = torch.tensor(np.array(len(data.x)), dtype=torch.float)
        test_n0_values.append(n0)
        SI = np.median(data.perimeter/np.sqrt(data.area))
        test_SI_values.append(SI)


    plt.figure(figsize=(6, 3))

    # plot mobility against cell number density
    plt.subplot(121)
    plt.scatter(train_n0_values, train_y_values, s=5, alpha=1.0, c='mediumaquamarine')
    plt.scatter(test_n0_values, test_y_values, s=5, alpha=1.0,c='salmon')

    # linear fit
    test_n0_values = torch.stack(test_n0_values).numpy()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_n0_values, test_y_values)
    predicted_y_values = slope * test_n0_values + intercept
    correlation = r_value
    mean_squared_error = np.mean((predicted_y_values - test_y_values) ** 2)
    plt.plot(test_n0_values, predicted_y_values, color='k', label='Fitted Line')
    print(f'Corr n0 (test) from fitting: {correlation:.4f}, mse n0 (test) from fitting: {mean_squared_error:.4f}')

    plt.xlabel('Cell number, N',fontsize=16)
    plt.ylabel('Mobility, M',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # shape index dependency plots
    plt.subplot(122)
    plt.scatter(train_SI_values, train_y_values, s=5, alpha=1.0, c='mediumaquamarine')
    plt.scatter(test_SI_values, test_y_values, s=5, alpha=1.0, c='salmon')

    # linear fitting
    test_SI_values = np.array(test_SI_values)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_SI_values, test_y_values)
    predicted_y_values = slope * test_SI_values + intercept
    correlation = r_value
    mean_squared_error = np.mean((predicted_y_values - test_y_values) ** 2)
    plt.plot(test_SI_values, predicted_y_values, color='k', label='Fitted Line')
    print(f'Corr SI (test) from fitting: {correlation:.4f}, mse SI (test) from fitting: {mean_squared_error:.4f}')

    plt.xlabel('$\overline{SI}$',fontsize=16)
    plt.ylabel('Mobility, M',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results/DataDistribution.png', dpi=300, bbox_inches='tight')

    # calculate correlation
    corr1, p_value = scipy.stats.pearsonr(np.array(train_n0_values),np.array(train_y_values))
    corr2, p_value = scipy.stats.pearsonr(np.array(train_SI_values),np.array(train_y_values))
    print(f'Corr n0 (train): {corr1:.4f}, Corr SI (train): {corr2:.4f}')

    corr1, p_value = scipy.stats.pearsonr(np.array(test_n0_values),np.array(test_y_values))
    corr2, p_value = scipy.stats.pearsonr(np.array(test_SI_values),np.array(test_y_values))
    print(f'Corr n0 (test): {corr1:.4f}, Corr SI (test): {corr2:.4f}')


# Function to show the Voronoi graphs
def visualizeDataGraph(data_list,ind_lst):
    for i in ind_lst:
        data = data_list[i]
        points= data.cell_pos
        area_lst=data.area
        peri_lst=data.perimeter
        SI_lst = peri_lst / np.sqrt(area_lst)
        voronoi_polygons = data.voronoi_polygons

        fig, ax = plt.subplots()

        edge_index_t=data.edge_index.T
        for edge in edge_index_t:
            plt.plot([points[edge[0], 0], points[edge[1], 0]],
                     [points[edge[0], 1], points[edge[1], 1]],
                     'k:', alpha=0.5)


        # colormap
        norm = mcolors.Normalize(vmin=3.7, vmax=4.5, clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)

        for polygon, SI in zip(voronoi_polygons, SI_lst):
            if polygon is not None:
              x, y = polygon.exterior.xy
              ax.fill(x, y, color=mapper.to_rgba(SI), edgecolor='white')

        plt.colorbar(mapper, ax=ax, orientation='vertical', label='Cell shape index')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/ExampleInputGraphs{i}.png', dpi=300, bbox_inches='tight')