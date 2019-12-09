import numpy as np

#Takes two datasets and does a hypothesis test based on the pearson correlation
def hypT_pearson(data1,data2,repnum):

    #Calculates the pearson r
    def pearson_r(x, y):
        """Compute Pearson correlation coefficient between two arrays."""
        # Compute correlation matrix: corr_mat
        corr_mat = np.corrcoef(x, y)

        # Return entry [0,1]
        return corr_mat[0,1]
    
    # Compute observed correlation: r_obs
    r_obs = pearson_r(data1, data2)

    # Initialize permutation replicates: perm_replicates
    perm_replicates = np.empty(repnum)

    # Draw replicates
    for i in range(len(perm_replicates)):
        # Permute data1 measurments: data1_permuted
        data1_permuted = np.random.permutation(data1)

        # Compute Pearson correlation
        perm_replicates[i] = pearson_r(data1_permuted, data2)
    
    # Compute p-value: p
    p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
    return p

