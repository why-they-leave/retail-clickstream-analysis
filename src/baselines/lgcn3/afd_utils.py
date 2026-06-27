import tensorflow as tf
# import numpy as np

def calculate_correlation(x):
    # x should be a 2D TensorFlow tensor
    x = tf.cast(x, tf.float32) # n*d

    # Step 1: Compute the covariance matrix
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True) # mean by columns
    x_centered = x - mean_x
    cov_matrix = tf.matmul(x_centered, x_centered, transpose_a=True) / tf.cast(tf.shape(x)[0] - 1, tf.float32)
    
    # Step 2: Compute the standard deviation of each variable
    stddev = tf.sqrt(tf.linalg.diag_part(cov_matrix)) # retrieves the diagnal part, all x-\bar{x}
    
    # Step 3: Compute the correlation matrix
    stddev_matrix = tf.matmul(tf.reshape(stddev, (-1, 1)), tf.reshape(stddev, (1, -1)))
    corr_matrix = cov_matrix / stddev_matrix # step1-3 should align with torch.corrcoef()
    
    # Step 4: Extract the upper triangular part of the matrix (excluding the diagonal)
    # triu_indices = np.triu_indices_from(np.zeros_like(corr_matrix), k=1) # this step can use numpy since constants
    # the k=1 means ignores the diagnal
    # triu_values = tf.gather_nd(corr_matrix, list(zip(triu_indices[0], triu_indices[1])))
    
    upper_triangular = tf.linalg.band_part(corr_matrix, 0, -1) - tf.linalg.band_part(corr_matrix, 0, 0)
    triu_values = tf.boolean_mask(upper_triangular, tf.not_equal(upper_triangular, 0))
    
    # Step 5: Compute the Frobenius norm of the extracted upper triangular part
    frobenius_norm = tf.norm(triu_values)
    
    return frobenius_norm # scalar
