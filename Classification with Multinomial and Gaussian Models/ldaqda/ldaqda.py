import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male, mu_female, cov, cov_male, cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    # Initialize the variables
    male_count = 0
    female_count = 0
    total_count = len(y)
    
    male_heights, male_weights = [], []
    female_heights, female_weights = [], []
    
    sum_male_heights, sum_male_weights = 0, 0
    sum_female_heights, sum_female_weights = 0, 0
    
    cov_male = np.array([[0,0], [0,0]])
    cov_female = np.array([[0,0], [0,0]])
    cov = np.array([[0,0], [0,0]])
    
    # Fill the variables
    for person in range(total_count):
        if y[person] == 1:
            male_count += 1
            male_heights.append(x[person][0])
            sum_male_heights += x[person][0]
            male_weights.append(x[person][1])
            sum_male_weights += x[person][1]
            
        else:
            female_count += 1
            female_heights.append(x[person][0])
            sum_female_heights += x[person][0]
            female_weights.append(x[person][1])
            sum_female_weights += x[person][1]
    
    # Do the calculations
    mu_male = [sum_male_heights/male_count, sum_male_weights/male_count]
    mu_female = [sum_female_heights/female_count, sum_female_weights/female_count]
    
    for i in range(x.shape[0]):
        all_data = np.array(x[i,:])
        if y[i] == 1:
            all_data = all_data - mu_male
            cov_male = np.add(cov_male, np.outer(all_data, all_data))
            
        else:
            all_data = all_data - mu_female
            cov_female = np.add(cov_female, np.outer(all_data, all_data))
            
        cov = np.add(cov, np.outer(all_data, all_data))
    
    cov_male = np.divide(cov_male, male_count)
    cov_female = np.divide(cov_female, female_count)
    cov = np.divide(cov, total_count)
    
    # Plotting LDA and QDA
    
    # Initialize the variables
    male_LDA, male_QDA = [], []
    female_LDA, female_QDA = [], []
    
    # Configure plot
    heights = np.linspace(50, 80, 100)   
    weights = np.linspace(80, 280, 100)
    
    X, Y = np.meshgrid(heights, weights)
    
    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    
    x_value = X[0].reshape(100, 1)
    
    for i in range(100):
      y_value = Y[i].reshape(100, 1)
      samples = np.concatenate((x_value, y_value), 1)
      
      male_LDA.append(util.density_Gaussian(mu_male, cov, samples))
      male_QDA.append(util.density_Gaussian(mu_male, cov_male, samples))

      female_LDA.append(util.density_Gaussian(mu_female, cov, samples))
      female_QDA.append(util.density_Gaussian(mu_female, cov_female, samples))   
    
    # Plot LDA
    plt.scatter(male_heights, male_weights, color='blue')
    plt.scatter(female_heights, female_weights, color='red') 
    
    # Plot the contours
    plt.contour(X, Y, male_LDA, colors='blue')
    plt.contour(X, Y, female_LDA, colors='red')  
    
    # Plot the boundaries
    LDA_boundary = np.asarray(male_LDA) - np.asarray(female_LDA)
    plt.contour(X, Y, LDA_boundary, 0, colors='red')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('LDA Plot')
    plt.savefig("lda.pdf")
    plt.show()
    
    # Plot QDA
    plt.scatter(male_heights, male_weights, color='blue')
    plt.scatter(female_heights, female_weights, color='red') 
    
    # Plot the contours
    plt.contour(X, Y, male_QDA, colors='blue')
    plt.contour(X, Y, female_QDA, colors='red')  
    
    # Plot the boundaries
    QDA_boundary = np.asarray(male_QDA) - np.asarray(female_QDA)
    plt.contour(X, Y, QDA_boundary, 0, colors='red')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('QDA Plot')
    plt.savefig("qda.pdf")
    plt.show()
    
    return (mu_male, mu_female, cov, cov_male, cov_female)
    

def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male, mu_female, cov, cov_male, mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    
    # Initialize the variables
    total_count = len(y)
    
    incorrect_LDA_count, incorrect_QDA_count = 0, 0
    
    cov_male_inverse = np.linalg.inv(cov_male)
    cov_female_inverse = np.linalg.inv(cov_female)
    cov_inv = np.linalg.inv(cov)
    
    mu_male_transpose = np.transpose(mu_male)
    mu_femalee_transpose = np.transpose(mu_female)
    
    
    # Calculate the constants
    alpha_male = - 0.5 * np.dot(np.dot(mu_male_transpose, cov_inv), mu_male)
    alpha_female = - 0.5 * np.dot(np.dot(mu_femalee_transpose, cov_inv), mu_female)

    beta_male = np.dot(mu_male_transpose, cov_inv)
    beta_female = np.dot(mu_femalee_transpose, cov_inv)
   
    # Calculate the misclassification rate in LDA
    for i, x_value in enumerate(x):
        male_LDA = np.dot(beta_male, x_value.T) + alpha_male
        female_LDA = np.dot(beta_female, x_value.T) + alpha_female

        if male_LDA < female_LDA and y[i] == 1:
            incorrect_LDA_count += 1
            
        elif female_LDA < male_LDA and y[i] == 2:
            incorrect_LDA_count += 1

    mis_lda = incorrect_LDA_count / total_count
    
    # Calculate the misclassification rate in QDA
    for i, x_value in enumerate(x):
        male_QDA = - np.log(np.linalg.det(cov_male)) - np.dot(x_value, np.dot(cov_male_inverse, x_value.T)) + 2 * np.dot(mu_male_transpose, 
                    np.dot(cov_male_inverse.T, x_value.T)) - np.dot(mu_male_transpose, np.dot(cov_male_inverse, mu_male))

        female_QDA = - np.log(np.linalg.det(cov_female)) - np.dot(x_value, np.dot(cov_female_inverse, x_value.T)) + 2 * np.dot(mu_femalee_transpose, 
                    np.dot(cov_female_inverse.T, x_value.T)) - np.dot(mu_femalee_transpose, np.dot(cov_female_inverse, mu_female))

        if male_QDA < female_QDA and y[i] == 1:
            incorrect_QDA_count += 1
            
        elif female_QDA < male_QDA and y[i] == 2:
            incorrect_QDA_count += 1

    mis_qda = incorrect_QDA_count / total_count
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # Load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # Parameter estimation and visualization in LDA/QDA
    mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train, y_train)
    
    # Misclassification rate computation
    mis_LDA, mis_QDA = misRate(mu_male, mu_female, cov, cov_male, cov_female, x_test, y_test)
    print("mis_LDA", mis_LDA)
    print("mis_QDA", mis_QDA)