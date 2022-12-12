import numpy as np
import math

# Eduidance distance
def dist(a, b):
    distance = 0
    for i in range(a.shape[0]):
        distance += (a[i]-b[i])*(a[i]-b[i])

    return distance


def K_Mean(centers, data, clusters, iterations, threshold):
    #Calculate mean of the whole dataset
    mean = np.mean(data, axis=0)
    # Create a list to store the SSE of each iterations
    sum_errors = [99] * iterations

    # #A value to store the improvement of SSE in each iterations
    error_improve = math.inf

    # Create the the 'distance' column to riginal data to indincate the distance of the instance to their cluster center, first column of the array
    data = np.c_[[math.inf] * data.shape[0], data]
    #
    # #A value to keep track of the iteration step (Which iteration it is)
    step = 0

    # #Let jump into the while loop
    # #The while loop will exit if the improvment of SSE small than threshold, or running out of the iterations, or there is no improvememnt in SSE
    while ((error_improve - threshold >= 0) and (iterations - step > 0)):
        # Reset the distance to infinite
        data[:, 0] = [math.inf] * data.shape[0]
        # Calculate all the distance between all the instances to all the cluster centers
        for i in range(data.shape[0]):
            for j in range(clusters):
                # Eduidance distance
                dis = dist(data[i, 2:data.shape[1]], centers[j, :])
                # Store the distance to the cluster_dis dataframe
                # Assign the instance to the closest cluster
                if (dis < data[i][0]):
                    # assign j to the cluster
                    data[i][1] = j
                    # assign closest distance
                    data[i][0] = dis

        data_cluster_errs = np.zeros((clusters, data.shape[1] + 1))
        for i in range(data.shape[0]):
            cluster = int(data[i, 1])
            data_cluster_errs[cluster, 0] += data[i, 0]
            data_cluster_errs[cluster, 1] = cluster
            data_cluster_errs[cluster, 2] += 1
            data_cluster_errs[cluster, 3:data.shape[1] + 1] += data[i, 2:data.shape[1]]

        # Handling empty cluster
        # Check if there is any 0 in the third column of data_cluster errs
        if (np.isin(0, data_cluster_errs[:, 2]).any()) == True:
            for i in range(clusters):
                # Pass if the cluster is not empty(the count is not 0)
                if (np.isin(0, data_cluster_errs[i, 2]).all()) == False:
                    pass
                else:
                    # Sort the data based on the distance with ascend order, then get the last index values, which contributes the most to the overall SSE
                    data = data[np.argsort(data[:, 0])]
                    # Change the cluster name to be the empty cluster
                    data[-1, 1] = i
                    # Change the distance to center to 0
                    data[-1, 0] = 0
            data_cluster_errs = np.zeros((clusters, data.shape[1] + 1))
            for i in range(data.shape[0]):
                cluster = int(data[i, 1])
                data_cluster_errs[cluster, 0] += data[i, 0]
                data_cluster_errs[cluster, 1] = cluster
                data_cluster_errs[cluster, 2] += 1
                data_cluster_errs[cluster, 3:data.shape[1] + 1] += data[i, 2:data.shape[1]]
        # Get the sum of Sum-of-Squared Error
        sum_errors[step] = data_cluster_errs[:, 0].sum()

        # if step == 0:
            # Get the initial SSE
            # print("Initial SSE = {:.4f}".format(sum_errors[step])+"; " +"delta SSE = 0")
            # Skip this part if it is first step since sse[0] = infnite
        if step >= 1:
            # Update the improvement in SSE
            error_improve = (sum_errors[step - 1] - sum_errors[step]) / sum_errors[step - 1]
        #    #Print result
        #     print("Iteration {}: SSE = {:.4f}".format(step + 1, sum_errors[step]) + "; " +"delta SSE = {:}".format(error_improve))

        # file.write("Iteration {}: SSE = {:.4f}".format(step+1,sum_errors[step])+'\n')

        # Exit the while loop if there is no improvement in SSE
        if error_improve == 0:
            # print("Final SSE = {:.4f}".format(sum_errors[step]))
            # print('Number of Iterations: {}'.format(step))
            # ch_k = CH(mean, data_cluster_errs[:, 1:3], centers, sum_errors[step - 1])
            # sw_k = SW(data, centers,data_cluster_errs[:,2])
            # db_k = DB(data, data_cluster_errs)
            return sum_errors[step], mean,data_cluster_errs, centers, data
        else:
            # Assign the new centers for next iterations
            for i in range(clusters):
                centers[i] = data_cluster_errs[i, 3:data.shape[1] + 1] / data_cluster_errs[i, 2]
            step += 1
    # print("Final SSE = {:.4f}".format(sum_errors[step - 1]))
    # print('Number of Iterations: {}'.format(step))
    # Return the lastest SSE
    #return sum_errors[step - 1]
    # ch_k = CH(mean,data_cluster_errs[:,1:3], centers, sum_errors[step - 1])
    # sw_k = SW(data, centers,data_cluster_errs[:,2])
    # db_k = DB(data, data_cluster_errs)
    return sum_errors[step - 1], mean,data_cluster_errs, centers, data
#Davies–Bould
def DB(data, data_cluster_errs):
    means = np.array([data_cluster_errs[i, 3:data.shape[1] + 1] / data_cluster_errs[i, 2] for i in range(data_cluster_errs.shape[0])])
    stds = np.zeros((data_cluster_errs.shape[0],))
    for i in range(data.shape[0]):
        for j in range(data_cluster_errs.shape[0]):
            if data[i,1] == j:
                stds[j] += dist(data[i, 2:data.shape[0]], means[j,:])*dist(data[i, 2:data.shape[0]], means[j,:])
    for counts in range(stds.shape[0]):
        stds[counts] =  np.sqrt(stds[counts] / data_cluster_errs[counts, 2])

    dbs = 0
    for i in range(stds.shape[0]):
        db = 0
        for j in range(0, stds.shape[0]):
            if(i != j):
                db_i = (stds[i]+stds[j])/(dist(means[i],means[j]))
                if (db_i>db):
                    db = db_i
        dbs += db
    dbs = dbs/stds.shape[0]

    return dbs


# #Silhouette Coefficient/Width
def SW(data,centers,cluster_count):
    # Add a column to keep track of the mean distance from xi to points in its own cluster y.
    data = np.c_[[0] * data.shape[0], data]
    # Add a column to keep track of the mean of the distances from xi to points in the closest cluster
    data = np.c_[[0] * data.shape[0], data]

    # print(data[:,0])
    # data = data[np.argsort(data[:,3])]
    # print(data[:,3])
    # print(data[0])
    # print(data[0, 4:data.shape[1]])
    #cluster_counts = dict(zip(unique, counts))
    # print(cluster_counts)
    for i in range(data.shape[0]):
        dis_ic = math.inf
        closest_cluster = 0
        for c in range(centers.shape[0]):
            dis = dist(data[i, 4:data.shape[1]], centers[c])
            if c!= data[i,3]:
                if dis < dis_ic:
                    dis_ic = dis
                    closest_cluster = c
        for j in range(data.shape[0]):
            if i != j:
                if data[i,3] == data[j,3]:
                    data[i,1] += dist(data[i, 4:data.shape[1]],data[j, 4:data.shape[1]])
                if data[j,3] == closest_cluster:
                    data[i,0] += dist(data[i, 4:data.shape[1]],data[j, 4:data.shape[1]])
        data[i,0]= data[i,0]/cluster_count[closest_cluster]
        data[i,1] = data[i,1]/cluster_count[int(data[i,3])]


    # print(data[:,0])
    # print(data[:,1])
    si = 0
    for i in range(data.shape[0]):
        si += (data[i,0]- data[i,1])/max(data[i,0], data[i,1])
    si = si/data.shape[0]

    return si

#Calinski–Harabasz Index
def CH(data_mean, cluster_count, centers, final_sse):
    data_mean = data_mean[1: len(data_mean)]
    num_points = np.sum(cluster_count[:,1])
    sb_sum = 0
    for i in range(cluster_count.shape[0]):
        sb_sum += cluster_count[i, 1] * np.dot((centers[i] - data_mean), (centers[i] - data_mean).T)
    ch_k = ((num_points-cluster_count.shape[0])*sb_sum)/((cluster_count.shape[0]-1)*final_sse)

    return ch_k

