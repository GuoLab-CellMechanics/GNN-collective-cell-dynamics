import os
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon
import matplotlib.colors as mcolors
import scipy.stats
import scipy.io



'''
Normalization
'''
def getLengthScale(directory):
    #
    foldernames = []
    for foldername in os.listdir(directory):
        if not foldername.startswith('.'):
            print(foldername)
            foldernames.append(os.path.join(directory, foldername))
    
    filenames = []
    for foldername in foldernames:
        for filename in os.listdir(foldername):
            if not filename.startswith('.'):
                filenames.append(os.path.join(foldername, filename))
                print(filename)

    print(len(filenames))
    N_lst=[]
    for filename in filenames:
        matdata = scipy.io.loadmat(filename)
        N_lst.append(len(matdata['cy']))
    
    # calculate lc
    N_avg = np.array(N_lst).mean()
    Area_tot = 691*516 # pixel^2
    Area_avg = Area_tot/N_avg
    lc = np.sqrt(Area_avg)
    return lc

# calculate the length scale for normalization
lc = getLengthScale(directory = "processed")
print('characteritic length for normalization',lc)
print('normalized x dimension', 691/lc)
print('normalized y dimension',516/lc)


'''
Construct dataset
'''
# tessellation
def calculate_Voronoi(points):
    vor = Voronoi(points)
    area_lst = np.zeros(len(points))
    perimeter_lst = np.zeros(len(points))
    is_within_threshold_lst = np.ones(len(points), dtype=bool)
    voronoi_polygons = [None] * len(points)  # Initialize with None or empty polygons

    for point_idx, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 not in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            poly_shape = Polygon(polygon)
            if poly_shape.is_valid:
                area = poly_shape.area
                perimeter = poly_shape.length

                # Check if any vertex is outside the ROI
                polygon_array = np.array(polygon)
                
                # Select ROI : x 2-30, y 2-22
                x_min = 2.0
                x_max = 30.0
                y_min = 2.0
                y_max = 22.0
                if np.all(polygon_array[:,0] > x_min) and np.all(polygon_array[:,0] < x_max)\
                        and np.all(polygon_array[:,1] > y_min) and np.all(polygon_array[:,1] < y_max):
                    area_lst[point_idx] = area
                    perimeter_lst[point_idx] = perimeter
                    voronoi_polygons[point_idx] = poly_shape  # Store the valid polygon
                else:
                    is_within_threshold_lst[point_idx] = False
            else:
                is_within_threshold_lst[point_idx] = False
        else:
            is_within_threshold_lst[point_idx] = False

    return area_lst, perimeter_lst, voronoi_polygons, is_within_threshold_lst

#
def Custom_dataset(filepath0, lc):
    
    # load matlab file
    matdata = scipy.io.loadmat(filepath0)
    
    # Create node_sublist
    x_coords = matdata['cx']/lc
    y_coords = matdata['cy']/lc
    node_sublist = np.hstack((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)))
    
    # Delaunay
    tri = Delaunay(node_sublist)
    edge_list = tri.simplices
    edges = set()
    for simplex in tri.simplices:
        # Extract vertices
        v1, v2, v3 = simplex
        # Create edges
        edges.add((v1, v2))
        edges.add((v2, v3))
        edges.add((v3, v1))
    
    # Voronoi and filter
    area_lst, perimeter_lst, voronoi_polygons, is_within_threshold_lst = calculate_Voronoi(node_sublist)
    filtered_area_lst = area_lst[is_within_threshold_lst]
    filtered_perimeter_lst = perimeter_lst[is_within_threshold_lst]
    filtered_node_sublist = node_sublist[is_within_threshold_lst]
    filtered_voronoi_polygons = [polygon for i, polygon in enumerate(voronoi_polygons) if is_within_threshold_lst[i]]

    # collect nodal and bulk information
    node_input = np.hstack((filtered_area_lst.reshape(-1, 1), filtered_perimeter_lst.reshape(-1, 1)))
    x = torch.tensor(node_input, dtype=torch.float)# input
    cell_pos = torch.tensor(filtered_node_sublist, dtype=torch.float)
    
    # output
    meanDisp = torch.tensor(matdata['meanDisp'], dtype=torch.float)/lc
    
    # filter edge
    filtered_indices = np.where(is_within_threshold_lst)[0]
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(filtered_indices)}
    filtered_edges = set()
    for edge in edges:
        v1, v2 = edge
        if v1 in index_mapping and v2 in index_mapping:
            new_v1 = index_mapping[v1]
            new_v2 = index_mapping[v2]
            filtered_edges.add((new_v1, new_v2))

    # get edge list
    edge_list = np.array(list(filtered_edges))
    edge_list = np.transpose(edge_list)

    edge_index = torch.tensor(edge_list, dtype=torch.long)

    edge_attr = ((filtered_node_sublist[edge_index[0,:],0] - filtered_node_sublist[edge_index[1,:],0])**2\
                +(filtered_node_sublist[edge_index[0,:],1] - filtered_node_sublist[edge_index[1,:],1])**2)**0.5
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # save data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, cell_pos = cell_pos,
                meanDisp = meanDisp,
                voronoi_polygons = filtered_voronoi_polygons)

    return data


#
def MainGetData(directory,lc):
    foldernames = []
    for foldername in os.listdir(directory):
        if not foldername.startswith('.'):
            print(foldername)
            foldernames.append(os.path.join(directory, foldername))

    pos_data_list = []
    for foldername in foldernames:
        # get all the filenames in a folder
        filenames = []
        for filename in os.listdir(foldername):
            if not filename.startswith('.'):
                filenames.append(os.path.join(foldername, filename))

        # Append the data
        data_list = []
        for filename in filenames:
            print(filename)
            data = Custom_dataset(filename,lc)
            data_list.append(data)

        pos_data_list.append(data_list)
    
    return pos_data_list

# The full dataset
data_list_all = MainGetData(directory = "processed", lc=lc)


'''
Visualization
'''
def visualizeDataGraph(data_list,ind_lst):
    for i in ind_lst:
        data = data_list[i]
        points= data.cell_pos
        area_lst=data.x[:,0]
        voronoi_polygons = data.voronoi_polygons

        # plot
        fig, ax = plt.subplots()

        edge_index_t=data.edge_index.T
        for edge in edge_index_t:
            plt.plot([points[edge[0], 0], points[edge[1], 0]],
                     [points[edge[0], 1], points[edge[1], 1]],
                     'k:', alpha=0.5)


        # colormap
        norm = mcolors.Normalize(vmin=min(area_lst), vmax=max(area_lst), clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)

        for polygon, area in zip(voronoi_polygons, area_lst):
            if polygon is not None:
                x, y = polygon.exterior.xy  # Get the coordinates of polygon vertices
                ax.fill(x, y, color=mapper.to_rgba(area), edgecolor='white')

        #
        plt.colorbar(mapper, ax=ax, orientation='vertical', label='Area')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        # plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

# visualize
visualizeDataGraph(data_list_all[0], np.linspace(0, len(data_list_all[0])-1, num=10, endpoint=True, dtype=int))


'''
Uncomment to save data
'''
# import pickle
# def save_dataset(data_list, file_name):
#     with open(file_name, 'wb') as f:
#         pickle.dump(data_list, f)

# savePath = "AllExpData.pkl"
# save_dataset(data_list_all, savePath)