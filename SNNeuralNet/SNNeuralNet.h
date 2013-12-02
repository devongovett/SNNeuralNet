//
//  SNNeuralNet.h
//  SNNeuralNet
//
//  Created by Devon Govett on 12/1/13.
//  Copyright (c) 2013 Devon Govett. All rights reserved.
//

// Holds a single training record
typedef struct {
    double *input;
    double *output;
} SNTrainingRecord;

// Useful for constructing input and output arrays with a nicer syntax
#define SNInput(...) (double[]){__VA_ARGS__}
#define SNOutput(...) SNInput(__VA_ARGS__)

// The actual neural network class
@interface SNNeuralNet : NSObject <NSCoding>

// Initializes an SNNeuralNet with a number of inputs and outputs
- (instancetype)initWithInputs:(int)numInputs outputs:(int)numOutputs;

// Initializes an SNNeuralNet with a number of inputs, an array of
// hidden layer sizes, and a number of outputs
- (instancetype)initWithInputs:(int)numInputs
                  hiddenLayers:(NSArray *)hiddenLayers
                       outputs:(int)numOutputs;

// Initializes and trains an SNNeuralNet in one step
- (instancetype)initWithTrainingData:(SNTrainingRecord *)trainingData
                          numRecords:(int)records
                           numInputs:(int)numInputs
                          numOutputs:(int)numOutputs;

// Trains the neural network with a set of training data.
// Returns the amount of training error that occurred.
// The neural network can only be trained once, and will
// return -1 if you attempt to train it multiple times.
- (double)train:(SNTrainingRecord *)trainingData numRecords:(int)records;

// Runs the neural network on an array of inputs
// of length numInputs
- (double *)runInput:(double *)input;

// Configurable properties
@property (nonatomic) int maxIterations;
@property (nonatomic) double minError;
@property (nonatomic) double learningRate;
@property (nonatomic) double momentum;

// State the neural network was created with
@property (readonly) int numInputs;
@property (readonly) int numOutputs;
@property (readonly) NSArray *hiddenLayers;

// Whether the neural network has been trained.
// It can only be trained once.
@property (readonly) BOOL isTrained;

@end
