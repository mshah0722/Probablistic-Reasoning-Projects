import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    
    # Get the file lists
    spam_file_lists = file_lists_by_category[0]
    ham_file_lists = file_lists_by_category[1]
    all_file_lists = spam_file_lists + ham_file_lists
    
    # Get the word frequency in each file list
    spam_word_frequency = util.get_word_freq(spam_file_lists)
    ham_word_frequency = util.get_word_freq(ham_file_lists)
    all_word_frequency = util.get_word_freq(all_file_lists)
    
    # Get the total word count in each file list
    spam_word_count = sum(spam_word_frequency.values())
    ham_word_count = sum(ham_word_frequency.values())
    
    # Get the total count of distinct words
    distinct_total_word_count = len(all_word_frequency)
    
    # Store the probablity distribution of each file list
    spam_distribution = {}
    ham_distribution = {}
    
    # Go through each word and calculate the probability distribution
    for word in all_word_frequency:
        word_count_spam = 0
        word_count_ham = 0
        
        if word in spam_word_frequency:
            word_count_spam = spam_word_frequency[word]
        
        if word in ham_word_frequency:
            word_count_ham = ham_word_frequency[word]
            
        spam_distribution[word] = (word_count_spam + 1) / (spam_word_count + distinct_total_word_count)
        ham_distribution[word] = (word_count_ham + 1) / (ham_word_count + distinct_total_word_count)
    
    probabilities_by_category = (spam_distribution, ham_distribution)   
    return probabilities_by_category

def classify_new_email(filename, probabilities_by_category, prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    spam_distribution = probabilities_by_category[0]
    ham_distribution = probabilities_by_category[1]
    
    # Get the log likelihood of each category
    spam_log_prob = np.log(prior_by_category[0])
    ham_log_prob = np.log(prior_by_category[1])
    
    # Get the email word frequency
    email_word_frequency = util.get_word_freq([filename])
    
    # Get the likelihood of spam and ham
    for key in email_word_frequency.keys():
        if key in spam_distribution:
            spam_log_prob += np.log(spam_distribution[key]) * email_word_frequency[key]
        if key in ham_distribution:
            ham_log_prob += np.log(ham_distribution[key]) * email_word_frequency[key]
    
    if spam_log_prob > ham_log_prob:
        classify_result = ("spam", [spam_log_prob, ham_log_prob])
    else:
        classify_result = ("ham", [spam_log_prob, ham_log_prob])            
    
    return classify_result

if __name__ == '__main__':
    
    # Folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # Generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # Prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # # Classify emails from testing set and measure the performance
    # for filename in (util.get_files_in_folder(test_folder)):
    #     # Classify
    #     label, log_posterior = classify_new_email(filename,
    #                                              probabilities_by_category,
    #                                              priors_by_category)
        
    #     # Measure performance (the filename indicates the true label)
    #     base = os.path.basename(filename)
    #     true_index = ('ham' in base) 
    #     guessed_index = (label == 'ham')
    #     performance_measures[int(true_index), int(guessed_index)] += 1

    # template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # # Correct counts are on the diagonal
    # correct = np.diag(performance_measures)
    # # Totals are obtained by summing across guessed labels
    # totals = np.sum(performance_measures, 1)
    # print(template % (correct[0], totals[0], correct[1], totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    type_1 = []
    type_2 = []
    
    # Declare the ratios
    ratios = [1E-10, 1E-5, 1E-3, 1E-1, 1, 5, 10, 50, 100, 1E10, 1E15]
    
    # Classify for each ratio
    for ratio in ratios:
        performance_measures = np.zeros([2,2])
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                        probabilities_by_category,
                                                        priors_by_category)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            
            # Get the log posterior probabilities
            log_posterior_spam = log_posterior[0]
            log_posterior_ham = log_posterior[1]

            if log_posterior_spam + np.log(ratio) > log_posterior_ham:
                label = "spam"
            else:
                label = "ham"
            
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1
        
        template="Current ratio = %f. You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        
        # Totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (ratio, correct[0], totals[0], correct[1], totals[1]))
        type_1.append(totals[0] - correct[0])
        type_2.append(totals[1] - correct[1])

    plt.plot(type_1, type_2)
    plt.xlabel('Count of Type 1 errors')
    plt.ylabel('Count of Type 2 errors')
    plt.title('Trade off of Type 1 errors vs. Type 2 errors')
    plt.savefig("nbc.pdf")
    plt.show()
 