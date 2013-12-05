//
//  SNNeuralNet.h
//  SNNeuralNet
//
//  Created by Devon Govett on 12/1/13.
//  Copyright (c) 2013 Devon Govett. All rights reserved.
//

/**
 *  Holds a single training record
 */
typedef struct {
    double *input;
    double *output;
} SNTrainingRecord;

/**
 *  Useful for constructing input arrays with a nicer syntax
 *
 *  @param ... list of doubles for input
 *  @return inline array of doubles to be put in an SNTrainingRecord
 */
#define SNInput(...) (double[]){__VA_ARGS__}

/**
 *  Useful for constructing output arrays with a nicer syntax
 *
 *  @param ... list of doubles for output
 *  @return inline array of doubles to be put in an SNTrainingRecord
 */
#define SNOutput(...) SNInput(__VA_ARGS__)

/**
Represents a neural network. After creating a network with a specified number of inputs,
outputs, and hidden layers, use the train:numRecords: method to train the network with an
array of SNTrainingRecord structures. Then use the runInput: method to predict output for
unknown inputs.

This example approximates the XOR function using a neural network:

    #import "SNNeuralNet.h"

    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(1)},
        {SNInput(1,0), SNOutput(1)},
        {SNInput(1,1), SNOutput(0)}
    };

    SNNeuralNet *net = [[SNNeuralNet alloc] initWithTrainingData:records numRecords:4 numInputs:2 numOutputs:1];

    double *output = [net runInput:SNInput(1, 0)];
    printf("%f\n", output[0]); // 0.987
*/
@interface SNNeuralNet : NSObject <NSCoding>

/// @name Initializing a neural network

/**
 *  Initializes an SNNeuralNet with a number of inputs and outputs, and one default hidden layer.
 *
 *  @param numInputs  Number of inputs to network
 *  @param numOutputs Number of outputs from network
 *
 *  @return An initialized SNNeuralNet
 */
- (instancetype)initWithInputs:(int)numInputs outputs:(int)numOutputs;

/**
 *  Initializes an SNNeuralNet with a number of inputs, an array of
 *  hidden layer sizes, and a number of outputs
 *
 *  @param numInputs    Number of inputs to network
 *  @param hiddenLayers Array of hidden layer sizes.
 *  @param numOutputs   Number of outputs from network
 *
 *  @return An initialized SNNeuralNet
 */
- (instancetype)initWithInputs:(int)numInputs
                  hiddenLayers:(NSArray *)hiddenLayers
                       outputs:(int)numOutputs;

/**
 *  Initializes and trains an SNNeuralNet in one step
 *
 *  @param trainingData Array of training records
 *  @param records      Number of records in trainingData
 *  @param numInputs    Number of inputs to network
 *  @param numOutputs   Number of outputs from network
 *
 *  @return An initialized and trained SNNeuralNet
 */
- (instancetype)initWithTrainingData:(SNTrainingRecord *)trainingData
                          numRecords:(int)records
                           numInputs:(int)numInputs
                          numOutputs:(int)numOutputs;

/// @name Neural network tasks

/**
 *  Trains the neural network with a set of training data.
 *
 *  @param trainingData Array of training records
 *  @param records      Number of records in trainingData
 *
 *  @return Returns the amount of training error that occurred.
 *          The neural network can only be trained once, and will
 *          return -1 if you attempt to train it multiple times.
 */
- (double)train:(SNTrainingRecord *)trainingData numRecords:(int)records;

/**
 *  Runs the neural network on an array of inputs of length numInputs
 *
 *  @param input Array of doubles to use as input to the neural network.
 *               Could be created with the SNInput macro
 *  @return Array of doubles with output of neural network
 */
- (double *)runInput:(double *)input;

/// @name Configuration

/**
 *  The maxium number of iterations to perform while training
 */
@property (nonatomic) int maxIterations;

/**
 *  The error threshold to reach while training, unless maxIterations is reached first
 */
@property (nonatomic) double minError;

/**
 *  The learning rate of the network
 */
@property (nonatomic) double learningRate;

/**
 *  Momentum of learning from previous inputs
 */
@property (nonatomic) double momentum;

/// @name Properties from initialization

/**
 *  Number of inputs the network was created with. Read only.
 */
@property (readonly) int numInputs;

/**
 *  Number of outputs the network was created with. Read only.
 */
@property (readonly) int numOutputs;

/**
 *  Array of hidden layer sizes the network was created with. Read only.
 */
@property (readonly) NSArray *hiddenLayers;

/// @name Other properties

/**
 *  Whether the neural network has been trained. It can only be trained once.
 */
@property (readonly) BOOL isTrained;

@end
