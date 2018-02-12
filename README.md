# convolutionnetwork

    THis is done on 200x200 training data set of holograms
    X are input of cropped holograms
    Y are a range of depths classified into 58 unique signatures
    
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 200,200,1)
    Y_train -- test set, of shape (None, n_y = 58)
    X_test -- training set, of shape (None, 200,200,1)
    Y_test -- test set, of shape (None, n_y = 58)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
