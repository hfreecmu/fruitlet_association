import os
import numpy as np
import networkx as nx
import distinctipy
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from inhand_utils import read_dict, extract_point_cloud
from inhand_utils import read_pickle, read_dict, create_point_cloud
from inhand_utils import write_dict

def process_cloud(segmentations, points, colors,
                  min_area, 
                  eps, eig_z_thresh,
                  vis, im):
    seg_points = []
    seg_colors = []
    centroids = []
    indices = []
    cluster_ids = {}
    for i in range(len(segmentations)):
        seg_inds = segmentations[i]
        det_seg_points = points[seg_inds[:, 0], seg_inds[:, 1]]
        nan_inds = np.isnan(det_seg_points).any(axis=1)
        filt_seg_inds = seg_inds[~nan_inds]

        if vis:
            im[seg_inds[:, 0], seg_inds[:, 1]] = [0, 0, 0]

        cluster_ids[i] = 'too_small_area'

        if filt_seg_inds.shape[0] < min_area:
            continue

        filt_points = points[filt_seg_inds[:, 0], filt_seg_inds[:, 1]]
        filt_colors = colors[filt_seg_inds[:, 0], filt_seg_inds[:, 1]]

        clustering = DBSCAN(eps=eps, min_samples=min_area).fit(filt_points)

        labels = clustering.labels_
        max_cluster_area = -1
        max_label_inds = None
        for label_id in np.unique(labels):
            if label_id == -1:
                continue

            label_inds = np.argwhere(labels == label_id)
            num_labels = label_inds.shape[0]
            if num_labels > max_cluster_area:
                max_cluster_area = num_labels
                max_label_inds = label_inds
        
        if max_label_inds is None:
            continue

        if max_cluster_area < min_area:
            continue

        filt_points = filt_points[max_label_inds[:, 0]]
        filt_colors = filt_colors[max_label_inds[:, 0]]

        pca = PCA(n_components=3)
        _ = pca.fit_transform(filt_points)
        _, eig_vecs = pca.explained_variance_, pca.components_
        
        cluster_ids[i] = 'bad_disparity'

        if np.abs(eig_vecs[0, 2]) > eig_z_thresh:
            if vis:
                im[seg_inds[:, 0], seg_inds[:, 1]] = [79,79,47]
            continue

        if vis:
            im[seg_inds[:, 0], seg_inds[:, 1]] = [255, 255, 255]

        cluster_ids[i] = 'unassigned'

        seg_points.append(filt_points)
        seg_colors.append(filt_colors)
        centroids.append(np.median(filt_points, axis=0))
        indices.append(i)

    centroids = np.array(centroids)
    indices = np.array(indices)

    return seg_points, seg_colors, centroids, indices, cluster_ids

def cluster_communities(centroids, indices, 
                        matched_indices, clusters, 
                        max_node_dist, 
                        weight_scale=100):

    G = nx.Graph()
    for ind_0 in range(len(indices)):
        if matched_indices[ind_0]:
            continue

        c_0 = centroids[ind_0]
        for ind_1 in range(ind_0 + 1, len(indices)):
            if matched_indices[ind_1]:
                continue

            c_1 = centroids[ind_1]

            dist = np.linalg.norm(c_0 - c_1)
            if dist > max_node_dist:
                continue
            
            weight = weight_scale*(-(dist/max_node_dist) + 1)
            G.add_edge(ind_0, ind_1, weight=weight)

    if len(G.nodes) == 0:
        communities = []
    else:
        communities = list(nx.community.louvain_communities(G))

    for _, _cluster in enumerate(communities):
        c = list(_cluster)

        cluster = []
        for ind in c:
            matched_indices[ind] = True
            cluster.append(ind)

        clusters.append(cluster)

def get_cluster_centroids(clusters, fruitlet_centroids):
    cluster_centroids = []
    for cluster in clusters:
        cluster_centroid = np.mean(fruitlet_centroids[cluster], axis=0)
        cluster_centroids.append(cluster_centroid)

    cluster_centroids = np.array(cluster_centroids)
    return cluster_centroids

def merge_clusters(clusters,
                   cluster_centroids,
                   fruitlet_centroids,
                   max_merge_dist,
                   max_cluster_size=2):
    merged = True
    while merged:
        merged = False
        #this will break if only one cluster so just continue if not
        if len(clusters) < 2:
            continue

        cluster_inds = np.arange(len(clusters))
        for cluster_ind in cluster_inds:
            cluster = clusters[cluster_ind]

            if len(cluster) > max_cluster_size:
                continue

            if len(cluster) < 2:
                raise RuntimeError('Did not expect cluster size less than 2')
            
            cluster_centroid = cluster_centroids[cluster_ind]
            
            #take second closest cluster (because first is itself)
            dists = np.linalg.norm(cluster_centroids - cluster_centroid, axis=1)
            closest_cluster_ind = np.argsort(dists)[1]
            closest_cluster_centroid = cluster_centroids[closest_cluster_ind]
            closest_cluster = clusters[closest_cluster_ind]

            #now calculate new cluster centroid
            numerator = closest_cluster_centroid*len(closest_cluster) + cluster_centroid*len(cluster)
            denom = len(closest_cluster) + len(cluster)
            new_cluster_centroid = numerator / denom
                                    
            #now check if every fruitlet centroid is within distance
            dist_0 = np.linalg.norm(fruitlet_centroids[cluster] - new_cluster_centroid, axis=1)
            dist_1 = np.linalg.norm(fruitlet_centroids[closest_cluster] - new_cluster_centroid, axis=1)

            #merge the clusters
            if np.max([dist_0.max(), dist_1.max()]) < max_merge_dist:
                #add to closest cluster
                for ind in cluster:
                    closest_cluster.append(ind)

                #delete now unecessary cluster
                del clusters[cluster_ind]

                #adjust closest cluster centroid
                cluster_centroids[closest_cluster_ind] = new_cluster_centroid

                #remove old cluster centroid
                cluster_centroids = np.delete(cluster_centroids, [cluster_ind], axis=0)

                #now we have to restart loop
                merged = True
                break

#TODO 
#could make this so that max_fruitlet_dist is increased
#if max distance from current cluster to current centroid
#is greater than max_fruitlet_dist
def add_single(clusters, cluster_centroids, fruitlet_centroids,
               indices, matched_indices, max_fruitlet_dist):
    added = True
    while added:
        added = False

        #iterate through all free fruitlets
        for ind in range(len(indices)):
            if matched_indices[ind]:
                continue
        
            centroid = fruitlet_centroids[ind]

            best_matched_cluster_ind = None
            best_matched_cluster_dist = 200000
            best_matched_cluster_centroid = None

            #iterate through all clusters
            cluster_inds = np.arange(len(clusters))
            for cluster_ind in cluster_inds:
                cluster = clusters[cluster_ind]
                cluster_centroid = cluster_centroids[cluster_ind]

                #now calculate new cluster centroid
                numerator = cluster_centroid*len(cluster) + centroid
                denom = len(cluster) + 1
                new_cluster_centroid = numerator / denom
                
                cluster_dist = np.linalg.norm(centroid - new_cluster_centroid)

                if cluster_dist >= max_fruitlet_dist:
                    continue

                if cluster_dist >= best_matched_cluster_dist:
                    continue

                #now check if every fruitlet centroid is within distance
                dist = np.linalg.norm(fruitlet_centroids[cluster] - new_cluster_centroid, axis=1)
                if dist.max() >= max_fruitlet_dist:
                    continue 

                best_matched_cluster_dist = cluster_dist
                best_matched_cluster_ind = cluster_ind
                best_matched_cluster_centroid = new_cluster_centroid

            if best_matched_cluster_ind is not None:
                cluster = clusters[best_matched_cluster_ind]
                
                #add fruitlet to cluster
                cluster.append(ind)
                #set matched to true
                matched_indices[ind] = True
                #update centroids
                cluster_centroids[best_matched_cluster_ind] = best_matched_cluster_centroid
                #now we have to restart loop
                added = True
                break

def vis_clusters(clusters, indices, segmentations, im, output_path):
    colors = distinctipy.get_colors(len(clusters))
    for cluster_ind in range(len(clusters)):
        cluster = clusters[cluster_ind]
        color = colors[cluster_ind]
        color = ([int(255*color[0]), int(255*color[1]), int(255*color[2])])
        for ind in cluster:
            seg_inds = segmentations[indices[ind]]
            im[seg_inds[:, 0], seg_inds[:, 1]] = color    

    cv2.imwrite(output_path, im)

def vis_cloud(seg_points, seg_colors, full_points, full_colors,
                 centroids, matched_indices, 
                 cloud_dir, basename):
    
    cloud_path = os.path.join(cloud_dir, basename.replace('.png', '.pcd'))
    create_point_cloud(cloud_path, np.vstack(seg_points), np.vstack(seg_colors))

    cloud_path = os.path.join(cloud_dir, basename.replace('.png', '_full.pcd'))
    create_point_cloud(cloud_path, full_points.reshape((-1, 3)), full_colors.reshape((-1, 3)))

    #do centroids and cluster centroids
    centroid_colors = []
    centroid_points = []
    for ind in range(len(centroids)):
        centroid_points.append(centroids[ind])
        if matched_indices[ind]:
            centroid_colors.append([0.0, 0.0, 1.0])
        else:
            centroid_colors.append([0.0, 1.0, 0.0])
    for cluster_centroid in cluster_centroids:
        centroid_points.append(cluster_centroid)
        centroid_colors.append([1.0, 0.0, 0.0])

    #centroid_points.append(tag_pos)
    #centroid_colors.append([1.0, 0.0, 1.0])

    centroid_points = np.array(centroid_points)
    centroid_colors = np.array(centroid_colors)
    cloud_path = cloud_path = os.path.join(cloud_dir, basename.replace('.png', '_centroids.pcd'))
    create_point_cloud(cloud_path, centroid_points, centroid_colors)


image_dir = '../preprocess_data/pair_images'
det_json_path = '../preprocess_data/pair_detections.json'
seg_dir = '../preprocess_data/pair_segmentations'
disparities_dir = '../preprocess_data/pair_disparities'
#will have detections and cluster id
output_dir = '../preprocess_data/pair_clusters'
camera_info_dir = '../preprocess_data/camera_info'
vis = True
vis_dir = '../preprocess_data/debug_cluster'
cloud_dir = '../preprocess_data/debug_clouds'
num_vis = 100

bilateral_filter = True
depth_discon_filter = True
distance_filter = True
min_area = 50
max_node_dist = 0.036
max_node_dist_2 = 0.05
max_merge_dist = 0.05
max_fruitlet_dist = 0.035
eig_z_thresh = 0.8
eps=0.01

#this is because we want to build up incrementally
#could adjust code to build up like we do with single fruitlet
assert max_node_dist <= max_node_dist_2

# start
image_dict = read_dict(det_json_path)
vis_count = 0
for basename in image_dict:
    image_path = os.path.join(image_dir, basename)
    disparity_path = os.path.join(disparities_dir, basename.replace('.png', '.npy'))
    seg_path = os.path.join(seg_dir, basename.replace('.png', '.pkl'))

    date_str = basename.split('_')[1]
    date_str = date_str.replace('-', '_')
    cam_info_path = os.path.join(camera_info_dir, date_str + '.yml')

    #TODO remove
    if not os.path.exists(disparity_path):
        continue

    points, full_colors = extract_point_cloud(image_path, disparity_path, cam_info_path,
                                           bilateral_filter, depth_discon_filter, 
                                           distance_filter)

    im = cv2.imread(image_path)
    segmentations = read_pickle(seg_path)

    seg_points, seg_colors, centroids, indices, cluster_ids = process_cloud(segmentations, points, full_colors, 
                                                               min_area,
                                                               eps, eig_z_thresh,
                                                               vis, im)
    
    
    matched_indices = [False]*indices.shape[0]
    clusters = []

    #fist pass at clustering
    cluster_communities(centroids, indices, matched_indices, clusters,
                        max_node_dist)
    
    if len(clusters) == 0:
        print('No clusters for: ' + basename)
        continue

    cluster_centroids = get_cluster_centroids(clusters, centroids)

    #merge clusters
    merge_clusters(clusters, cluster_centroids, centroids,
                   max_merge_dist)
    
    
    #second pass at clustering
    cluster_communities(centroids, indices, matched_indices, clusters,
                        max_node_dist_2)
    
    cluster_centroids = get_cluster_centroids(clusters, centroids)

    #second pass at merging
    merge_clusters(clusters, cluster_centroids, centroids, max_merge_dist)

    #lastly singles
    add_single(clusters, cluster_centroids, centroids, indices, 
               matched_indices, max_fruitlet_dist)
    
    #don't think I need this but just in case
    cluster_centroids = get_cluster_centroids(clusters, centroids)

    if (vis) and (vis_count < num_vis):
        vis_count += 1 
        vis_path = os.path.join(vis_dir, basename)
        vis_clusters(clusters, indices, segmentations, 
                     im, vis_path)
        
        vis_cloud(seg_points, seg_colors, points, full_colors,
                  centroids, matched_indices, cloud_dir, basename)
    
    #now save the dict
    num_clusters = len(clusters)
    cluster_dict = {}

    for cluster_ind in range(len(clusters)):
        cluster = clusters[cluster_ind]
        orig_inds = []
        for ind in cluster:
            orig_ind = int(indices[ind])
            if cluster_ids[orig_ind] != 'unassigned':
                raise RuntimeError('expected unassigned, debug this: ' + basename)
            
            cluster_ids[orig_ind] = cluster_ind
            orig_inds.append(orig_ind)

        cluster_dict[cluster_ind] = orig_inds

    
    output_dict = {"num_clusters": len(clusters),
                   "num_fruitlets": len(segmentations),
                   "clusters": cluster_dict,
                   "fruitlet_clusters": cluster_ids}
    
    output_path = os.path.join(output_dir, basename.replace('.png', '.json'))
    write_dict(output_path, output_dict)
