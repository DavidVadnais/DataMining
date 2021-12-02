# Document Classification
## Tools
* Check if dependencies are needed
    * https://docs.anaconda.com/anaconda/install/

## Data
http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/

## Related Works
See Kowsari, Kamran et al. “Text Classification Algorithms: A Survey.” Inf. 10 (2019): 150. for details https://github.com/kk7nc/Text_Classification#naive-bayes-classifier

# Assignment 
## Deliverables
The files expected in your submission folder are listed below. You will loose considerable points if you have less files
Project2Report-[lastname].pdf
20_newsgroups_Train(All Train data in this folder)
20_newsgroups_Test (All Test data in this folder)
eliminate.txt
BagofWords.txt
naive_bayes.[extension of language] 
neural_networks.[extension of language] 
support_vm.[extension of language] 
where: Project2Report-[lastname].pdf, contains the report of your results for the programming task 7. Only PDF files will be accepted. All text should be typed, and if any figures are present they should be computer-generated. Scans of handwriten answers will NOT be accepted.
Introduction
This classification problem involves classifying 20000 messages into 20 different classes using three classification algorithms Naïve Bayes, Neural Networks, and Support Vector Machine (SVM) for the training and test dataset. One of the most important aspect to solving this problem is having the appropriate data set format.

## Task 1: Data Download
Download the Twenty Newsgroups Data Set to answer the questions for this project.. Note: download only the 20_newsgroups.tar.gz file

20_newsgroups Dataset: The dataset contains 20,000 newsgroup messages drawn from the 20 newsgroups. The dataset contains 1000 documents from each of the 20 newsgroups. For classes descriptions, please refer Table 6.3 of Dr. Mitchell's book (Machine Learning, Tom Mitchell).

## Task 2: Create Training and Test Dataset (5 points)
In this task, create the training dataset and test dataset. Use 60% of the data as the training data and 40% of the data as the test data.
Name of Training dataset folder: 20_newsgroups_Train
Name of Test dataset folder: 20_newsgroups_Test

## Task 3: Create the Bag of Words (15 points)
As discussed in class, for this task you will create the dictionary or bag of words containing the unique words arranged in the order of based on the weight of the terms.
To Create the Bag of Words follow the following Steps:
Remove punctuations and common words(stop words) so that only the useful words will be remaining for classification
Compile all the removed words in a single text document called “eliminate.txt”
Perform a term-frequency count for each document, that is, the words in each document were also arranged based on popularity so that they can easily be indexed for further processing.
For each category (folder), create a group dictionary created by merging the documents frequency count as one.
Combine the group dictionaries (20 in this case) to form a dictionary so that there is only one dictionary (Bag of Words) for the whole set.
Now you should have the Bag of Words which is basically a popularity list of the words, the bag of words can also be termed dictionary. Save the Bag of Words in a single text document "BagofWords.txt"
Here is an example of the Bag of Words:
Bag of Words Example

## Task 4: Naive Bayes (NB) Algorithm(25 points)
In this task, you will implement a R, Matlab or Python function executable file, that uses Naive Bayes to train a model, and then applies the model to classify your test data.
Please implement the Naive Bayes classifier by yourself. Don't use any online code or Library. You lose the points for this question if you do otherwise as instructed.
Your code should be invoked as follows (Matlab):

				naive_bayes(<training_path>, <test_path>)
			
Your code should be invoked as follows (Python):

    		python3 naive_bayes.py <training_path>  <test_path> 
    	   
where:
* The first argument, <training_path>, is the path of the directory containing all the training data.
* The second argument, <test_path>, is the path of the directory containing all the test data
Code:
* naive_bayes.[extension of language] containing your R, Matlab or Python code for the programming part.
* In addition, you must include in the source files (with auxiliary code) that are needed to run your solution.
* Matlab code needs to run on version 2017a and later, and Python code needs to run on Anaconda version 3.6 (Python version 3.6.4, numpy version 1.13.3).

## Task 5: Neural Networks (12.5 points)
In this task, you will implement a R, Matlab or Python function executable file, that uses neural networks to train a model, and then applies the model to classify your test data.

Library: You can use any neural networks library to solve this

BONUS: If you implement a working version of Neural Networks with no library, you get 7 points bonus
Your code should be invoked as follows (Matlab):

				neural_networks(<training_path>, <test_path>)
			
Your code should be invoked as follows (Python):

    		python3 neural_networks.py <training_path>  <test_path> 
    	   
where:
* The first argument, <training_path>, is the path of the directory containing all the training data.
* The second argument, <test_path>, is the path of the directory containing all the test data
Code:
* neural_networks.[extension of language] containing your R, Matlab or Python code for the programming part.
* In addition, you must include in the source files or any auxiliary codes that are needed to run your solution.
* Matlab code needs to run on version 2017a and later, and Python code needs to run on Anaconda version 3.6 (Python version 3.6.4, numpy version 1.13.3).

## Task 6: Support Vector Machine SVM (12.5 points)
In this task, you will implement a R, Matlab or Python function executable file, that uses support vector machine to train a model, and then applies the model to classify your test data.

Library: You can use the libsvm library or any other SVM library to solve this
Your code should be invoked as follows (Matlab):

				support_vm(<training_path>, <test_path>)
			
Your code should be invoked as follows (Python):

    		python3 support_vm.py <training_path>  <test_path> 
    	   
where:
* The first argument, <training_path>, is the path of the directory containing all the training data.
* The second argument, <test_path>, is the path of the directory containing all the test data
Code:
* support_vm.[extension of language] containing your R, Matlab or Python code for the programming part.
* In addition, you must include in the source files or any auxiliary codes that are needed to run your solution.
* Matlab code needs to run on version 2017a and later, and Python code needs to run on Anaconda version 3.6 (Python version 3.6.4, numpy version 1.13.3).

## Task 7: Report (30 points)
<ol type="a">
  <li>Report the accuracy, number of misclassified, recall, and running time for the three algorithms. (15 points for undergrad/ 10 points for grads)</li>
  <li>What is the feature size you used for the Neural Networks and the SVM algorithm? (1 point)</li>
  <li>What is the name of the library you used to build for the Neural Networks and the SVM classifications? (1 point)</li>
  <li>Report the accuracy, number of misclassified, recall, and running time when the dictionary(Bag of Words) size is 70,000, 50,000, 30000, and 10000 for the Naive Bayes Algorithm. (10 points)</li>
  <li>Comment on the performance of (d) above and explain the reason for the increase or decrease in accuracy? (3 points )</li>
  <li>Reduce the feature size of neural Networks and the SVM algorithm by 10000 and report the accuracy, number of misclassified, recall, and running time . (5 points) (Gradate Students Only/Bonus for Undergrads)</li>
</ol>
