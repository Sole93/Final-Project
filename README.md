# pythonProject12
Final Project: Neural Network Program
Summary of Final Project

Explanation:
It is a simple artificial neural network program.

First of all, we built a class called NeuralNetwork,

in class first function is default function named __init__,

it will run when class object is called.

 

Then we create 4 more functions to train our model.

in train function there is just maths (derivations) formulas,

and in other than this function there is just one return statement.

 

In main function,

we open a file and read data from it.

Then we split data into input and output list,

for training of our model, we split data into train and test data using sklearn.train_test_split

then with model.train*() we train out dataset.


With sklearn.accuracy_score() we find the accuracy of our model.

 
And in the last, you can manually check prediction for our trained model.

For example, enter, **18 44 2 200 6**

it will answer 

Predicted result
0.0
Chance of fever
