SNNeuralNet
===============

A neural network library for Objective-C based on [brain.js](https://github.com/harthur/brain), for iOS and Mac OS X.

## Example

This example approximates the XOR function using a neural network:

```objective-c
#import "SNNeuralNet.h"

SNTrainingRecord records[] = {
    {SNInput(0,0), SNOutput(0)},
    {SNInput(0,1), SNOutput(1)},
    {SNInput(1,0), SNOutput(1)},
    {SNInput(1,1), SNOutput(0)}
};

SNNeuralNet *net = [[SNNeuralNet alloc] initWithTrainingData:records 
                                                  numRecords:4
                                                   numInputs:2
                                                  numOutputs:1];

double *output = [net runInput:SNInput(1, 0)];
printf("%f\n", output[0]); // 0.987
```

## Creating an SNNeuralNet

There are several ways of creating an `SNNeuralNet` instance. The base init method is:

```objective-c
SNNeuralNet *net = [[SNNeuralNet alloc] initWithInputs:2 outputs:1];
```

This will create a default neural network with one hidden layer, which is good enough for many
usecases. However, you may also create a neural network with more hidden layers with sizes of your
choosing.

```objective-c
SNNeuralNet *net = [[SNNeuralNet alloc] initWithInputs:2 hiddenLayers:@[@3, @4] outputs:1];
```

Additionally, you can create and train a network in one step, though this makes it impossible to 
configure some aspects of the network and create custom hidden layers.

```objective-c
// see the section on training below
SNTrainingRecord records[] = {
    {SNInput(0,0), SNOutput(0)},
    {SNInput(0,1), SNOutput(1)},
    {SNInput(1,0), SNOutput(1)},
    {SNInput(1,1), SNOutput(0)}
};

SNNeuralNet *net = [[SNNeuralNet alloc] initWithTrainingData:records 
                                                  numRecords:4
                                                   numInputs:2
                                                  numOutputs:1];
```

## Configuration

There are several configurable properties of an SNNeuralNet. Once you have an instance,
you can set these properties. These properties will have no effect after training, so make
sure to use one of the constructors that does not also do training. Their defaults are included
below.

```objective-c
net.maxIterations = 20000;  // maximum training iterations
net.minError = 0.005;       // error threshold to reach
net.learningRate = 0.3;     // influences how quickly the network trains
net.momentum = 0.1;         // influences learning rate
```

## Training

Once you've created and configured your network, you should train it with some known data. The 
network can only be trained once, so include all training data at once. If you attempt to call
the `train` method more than once, `-1` will be returned. You can check `net.isTrained` to see 
if a network has already been trained.

Before you can train, you need to make an array of `SNTrainingRecord`s.  These are C structures
that hold input and output arrays of doubles. You can have as many inputs or outputs as you like,
but all data must have the same number of inputs and outputs. There are convienience macros called
`SNInput` and `SNOutput` that wrap data in an array.

When you're ready to train the network, use the train method, and pass the record array and number of records.

```objective-c
SNTrainingRecord records[] = {
    {SNInput(0,0), SNOutput(0)},
    {SNInput(0,1), SNOutput(1)},
    {SNInput(1,0), SNOutput(1)},
    {SNInput(1,1), SNOutput(0)}
};

double error = [net train:records numRecords:4];
```

The `train` method returns the amount of error that occurred in training, which should be lower than 
`net.minError` unless `net.maxIterations` was reached.

## Running unknown inputs

The real usefulness of a neural network is its ability to predict outputs for unknown inputs. Once your
`SNNeuralNet` has been trained, you can use the `runInput` method to get predicted outputs.

```objective-c
double *output = [net runInput:SNInput(1, 0.4, 0)];
```

`runInput` returns an array of doubles, including `net.numOutputs` entries.

# License

MIT