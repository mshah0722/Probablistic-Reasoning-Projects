import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    # Store the parameters
    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    
    A_Mu = np.array([0, 0])
    A_Cov = np.array([[beta, 0], [0, beta]])
    
    # Plot the contours of the prior distribution
    X_label, Y_label = np.meshgrid(a0, a1)
    X_t = X_label[0].reshape(100, 1)
    Contour = []
    
    for i in range(100):
        Y_t = Y_label[i].reshape(100, 1)
        X_set = np.concatenate((X_t, Y_t), 1)
        Contour.append(util.density_Gaussian(A_Mu, A_Cov, X_set))
    
    graph1 = plt.figure(1)
    graph1_axes = graph1.add_axes([0.1, 0.1, 0.8, 0.8])
    A_value = plt.scatter(-0.1, -0.5, color = 'blue', s = 50)
    
    graph1_axes.set_xlim([-1, 1])
    graph1_axes.set_ylim([-1, 1])
    
    plt.contour(X_label, Y_label, Contour)
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution P(a)')  
    plt.legend([A_value], ['(a0, a1)'], scatterpoints = 1, loc = 'best')
    plt.show()
    # plt.savefig("prior.pdf")
    
    return 
    
def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    Mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    
    # Store the parameters
    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    
    n = len(x)
    X = np.append(np.ones((n, 1), dtype=int), x, 1)
    
    A_Cov = np.array([[beta, 0], [0, beta]])
    A_Cov_Inv = np.linalg.inv(A_Cov)
    
    W_Cov = np.array([sigma2])
    
    # Covariance and Mean of the posterior distribution
    Cov_Inv = A_Cov_Inv + (np.matmul(X.T, X) * 1/W_Cov)
    Cov = np.linalg.inv(Cov_Inv)
    
    Mu = 1/W_Cov * np.matmul(np.matmul(Cov, X.T), z)
    Mu = Mu.reshape(2, 1).squeeze()
    
    # Plot the contours of the prior distribution
    X_label, Y_label = np.meshgrid(a0, a1)
    X_t = X_label[0].reshape(100, 1)
    Contour = []
    
    for i in range(100):
        Y_t = Y_label[i].reshape(100, 1)
        X_set = np.concatenate((X_t, Y_t), 1)
        Contour.append(util.density_Gaussian(Mu, Cov, X_set))
    
    graph2 = plt.figure(2)
    graph2_axes = graph2.add_axes([0.1, 0.1, 0.8, 0.8])
    A_value = plt.scatter(-0.1, -0.5, color = 'blue', s = 50)
    
    graph2_axes.set_xlim([-1, 1])
    graph2_axes.set_ylim([-1, 1])
    
    plt.contour(X_label, Y_label, Contour)
    plt.xlabel('a0')
    plt.ylabel('a1')
    
    if n == 1:
        plt.title('Posterior 1: p(a|x, x1, z1)')
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
    if n == 5:
        plt.title('Posterior 5: p(a|x, x1, z1, ..., x5, z5)') 
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
    if n == 100:
        plt.title('Posterior 100: p(a|x, x1, z1, ..., x100, z100)')
        plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'blue')
        
    plt.show()
    # plt.savefig("posterior" + str(n) + ".pdf")
   
    return (Mu, Cov)

def predictionDistribution(x, beta, sigma2, Mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    Mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    # Store the parameters
    n = len(x_train)
    X = np.append(np.ones((len(x), 1), dtype=int), np.expand_dims(x, 1), 1)

    Z_Mu = np.dot(X, Mu)
    W_Cov = np.array([sigma2])
    Z_Cov = W_Cov + np.matmul(np.matmul(X, Cov), X.T)
    std_dev = np.sqrt(np.diag(Z_Cov))

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x')
    plt.ylabel('z')
    
    if n == 1:
        plt.title('Predict 1: p(z|x, x1, z1)')
    if n == 5:
        plt.title('Predict 5: p(z|x, x1, z1, ..., x5, z5)')
    if n == 100:
        plt.title('Predict 100: p(z|x, x1, z1, ..., x100, z100)')
    
    plt.errorbar(x, Z_Mu, yerr = std_dev, fmt = 'x', color = 'grey')
    plt.scatter(x_train, z_train, color = 'red', s=3)
    plt.show()   
    
    # plt.savefig("predict" + str(n) + ".pdf")
    return 

if __name__ == '__main__':
    
    # Training data
    x_train, z_train = util.get_data_in_file('training.txt')
    
    # New inputs for prediction 
    x_test = [x for x in np.arange(-4, 4.01, 0.2)]
    
    # Known parameters 
    sigma2 = 0.1
    beta = 1
    
    # Number of training samples used to compute posterior
    ns = 100
    
    # Used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # Prior distribution p(a)
    priorDistribution(beta)
    
    # Posterior distribution p(a|x,z)
    Mu, Cov = posteriorDistribution(x, z, beta, sigma2)
    
    # Distribution of the prediction
    predictionDistribution(x_test, beta, sigma2, Mu, Cov, x, z)
    