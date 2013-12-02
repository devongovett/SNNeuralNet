//
//  SNNeuralNet.m
//  SNNeuralNet
//
//  Created by Devon Govett on 12/1/13.
//  Copyright (c) 2013 Devon Govett. All rights reserved.
//

#import "SNNeuralNet.h"
#import <math.h>

typedef struct {
    double *deltas;
    double *errors;
    double *outputs;
    double *biases;
    double *weights;
    double *changes;
} SNLayer;

// option defaults
#define DEFAULT_MAX_ITERATIONS 20000
#define DEFAULT_MIN_ERROR 0.005
#define DEFAULT_LEARNING_RATE 0.3
#define DEFAULT_MOMENTUM 0.1

// allocation/initialization helpers
#define zeros(size) calloc(size, sizeof(double))
double *randos(int size)
{
    double *ptr = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
        ptr[i] = (double)arc4random() / 0x100000000;
    
    return ptr;
}

@implementation SNNeuralNet
{
    int *sizes;
    int outputLayer;
    SNLayer *layers;
}

- (instancetype)initWithInputs:(int)numInputs outputs:(int)numOutputs
{
    return [self initWithInputs:numInputs hiddenLayers:nil outputs:numOutputs];
}

- (instancetype)initWithInputs:(int)numInputs
                  hiddenLayers:(NSArray *)hiddenLayers
                       outputs:(int)numOutputs
{
    self = [super init];
    
    if (self) {
        // setup default options
        self.maxIterations = DEFAULT_MAX_ITERATIONS;
        self.minError = DEFAULT_MIN_ERROR;
        self.learningRate = DEFAULT_LEARNING_RATE;
        self.momentum = DEFAULT_MOMENTUM;
        
        // save configuration
        _numInputs = numInputs;
        _hiddenLayers = hiddenLayers;
        _numOutputs = numOutputs;
        
        // allocate layers and sizes
        outputLayer = 1 + (hiddenLayers != nil ? (int)hiddenLayers.count : 1);
        sizes = malloc((outputLayer + 1) * sizeof(int));
        layers = malloc((outputLayer + 1) * sizeof(SNLayer));
        
        // setup array of layer sizes
        sizes[0] = numInputs;
        if (hiddenLayers != nil) {
            for (int i = 0; i < hiddenLayers.count; i++)
                sizes[i + 1] = [hiddenLayers[i] intValue];
        } else {
            // one hidden layer by default
            sizes[1] = MAX(3, numInputs / 2);
        }
        
        sizes[outputLayer] = numOutputs;
        
        // allocate layers
        for (int layer = 0; layer <= outputLayer; layer++) {
            int size = sizes[layer];
            
            layers[layer].deltas = zeros(size);
            layers[layer].errors = zeros(size);
            layers[layer].outputs = zeros(size);
            
            if (layer > 0) {
                layers[layer].biases = randos(size);
                layers[layer].weights = randos(size * sizes[layer - 1]);
                layers[layer].changes = zeros(size * sizes[layer - 1]);
            }
        }
    }
    
    return self;
}

- (void)dealloc
{
    for (int layer = 0; layer <= outputLayer; layer++) {
        free(layers[layer].deltas);
        free(layers[layer].errors);
        free(layers[layer].outputs);
        
        if (layer > 0) {
            free(layers[layer].biases);
            free(layers[layer].weights);
            free(layers[layer].changes);
        }
    }
    
    free(layers);
    free(sizes);
}

- (instancetype)initWithTrainingData:(SNTrainingRecord *)trainingData
                          numRecords:(int)records
                           numInputs:(int)numInputs
                          numOutputs:(int)numOutputs
{
    self = [self initWithInputs:numInputs outputs:numOutputs];
    
    if (self) {
        [self train:trainingData numRecords:records];
    }
    
    return self;
}

- (double)train:(SNTrainingRecord *)trainingData numRecords:(int)records
{
    if (self.isTrained)
        return -1;
    
    double error = 1.0;

    for (int i = 0; i < self.maxIterations && error > self.minError; i++) {
        double sum = 0;
        for (int j = 0; j < records; j++)
            sum += [self trainPattern:&trainingData[j]];
        
        error = sum / records;
    }
    
    _isTrained = YES;
    return error;
}

- (double)trainPattern:(SNTrainingRecord *)record
{
    // forward propogate
    [self runInput:record->input];
    
    // back propogate
    [self calculateDeltas:record->output];
    [self adjustWeights];
    
    // mean squared error
    double *errors = layers[outputLayer].errors;
    double sum = 0;
    for (int i = 0; i < sizes[outputLayer]; i++) {
        sum += errors[i] * errors[i];
    }
    
    return sum / sizes[outputLayer];
}

- (double *)runInput:(double *)input
{
    memcpy(layers[0].outputs, input, sizes[0] * sizeof(double));
    
    for (int layer = 1; layer <= outputLayer; layer++) {
        for (int node = 0; node < sizes[layer]; node++) {
            int idx = node * sizes[layer - 1];
            double *weights = layers[layer].weights;
            double sum = layers[layer].biases[node];
            
            for (int k = 0; k < sizes[layer - 1]; k++) {
                sum += weights[idx + k] * input[k];
            }
            
            layers[layer].outputs[node] = 1 / (1 + exp(-sum));
        }
        
        input = layers[layer].outputs;
    }
    
    return layers[outputLayer].outputs;
}

- (void)calculateDeltas:(double *)target
{
    for (int layer = outputLayer; layer >= 0; layer--) {
        for (int node = 0; node < sizes[layer]; node++) {
            double output = layers[layer].outputs[node];
            
            double error = 0;
            if (layer == outputLayer) {
                error = target[node] - output;
            } else {
                double *deltas = layers[layer + 1].deltas;
                for (int k = 0; k < sizes[layer + 1]; k++) {
                    error += deltas[k] * layers[layer + 1].weights[node + k * sizes[layer]];
                }
            }
            
            layers[layer].errors[node] = error;
            layers[layer].deltas[node] = error * output * (1 - output);
        }
    }
}

- (void)adjustWeights
{
    for (int layer = 1; layer <= outputLayer; layer++) {
        double *incoming = layers[layer - 1].outputs;
        
        for (int node = 0; node < sizes[layer]; node++) {
            double delta = layers[layer].deltas[node];
            
            for (int k = 0; k < sizes[layer - 1]; k++) {
                int idx = k + node * sizes[layer - 1];
                double change = layers[layer].changes[idx];
                
                change = (self.learningRate * delta * incoming[k]) + (self.momentum * change);
                
                layers[layer].changes[idx] = change;
                layers[layer].weights[idx] += change;
            }
            
            layers[layer].biases[node] += self.learningRate * delta;
        }
    }
}

#pragma mark - NSCoding

// helper macros for initWithCoder and encodeWithCoder
#define ArrayFromNSData(key) {                   \
    NSData *data = dict[@"key"];                 \
    memcpy(layer->key, data.bytes, data.length); \
}

#define NSDataFromArray(data, size) [NSData dataWithBytes:data length:size * sizeof(double)]

- (instancetype)initWithCoder:(NSCoder *)decoder
{
    self = [self initWithInputs:[decoder decodeIntForKey:@"numInputs"]
                                            hiddenLayers:[decoder decodeObjectForKey:@"hiddenLayers"]
                                                 outputs:[decoder decodeIntForKey:@"numOutputs"]];
    
    if (self) {
        self.maxIterations = [decoder decodeIntForKey:@"maxIterations"];
        self.minError = [decoder decodeDoubleForKey:@"minError"];
        self.learningRate = [decoder decodeDoubleForKey:@"learningRate"];
        self.momentum = [decoder decodeDoubleForKey:@"momentum"];
        _isTrained = [decoder decodeBoolForKey:@"isTrained"];
        
        NSArray *layersArray = [decoder decodeObjectForKey:@"layers"];
        for (int i = 0; i <= outputLayer; i++) {
            NSDictionary *dict = layersArray[i];
            SNLayer *layer = &layers[i];
            
            // copy data from encoded layers to self
            ArrayFromNSData(deltas);
            ArrayFromNSData(errors);
            ArrayFromNSData(outputs);
            if (i > 0) {
                ArrayFromNSData(biases);
                ArrayFromNSData(weights);
                ArrayFromNSData(changes);
            }
        }
    }
    
    return self;
}

- (void)encodeWithCoder:(NSCoder *)encoder
{
    // encode each layer as a dictionary, with each key as NSData
    NSMutableArray *layersArray = [NSMutableArray array];
    
    for (int i = 0; i <= outputLayer; i++) {
        SNLayer *layer = &layers[i];
        int size = sizes[i];
        
        NSMutableDictionary *dictionary = [NSMutableDictionary dictionaryWithDictionary:@{
            @"deltas": NSDataFromArray(layer->deltas, size),
            @"errors": NSDataFromArray(layer->errors, size),
            @"outputs": NSDataFromArray(layer->outputs, size),
        }];
        
        if (i > 0) {
            dictionary[@"biases"] = NSDataFromArray(layer->weights, size);
            dictionary[@"weights"] = NSDataFromArray(layer->weights, size * sizes[i - 1]);
            dictionary[@"changes"] = NSDataFromArray(layer->changes, size * sizes[i - 1]);
        }
        
        [layersArray addObject:dictionary];
    }
    
    [encoder encodeInt:self.numInputs forKey:@"numInputs"];
    [encoder encodeObject:self.hiddenLayers forKey:@"hiddenLayers"];
    [encoder encodeInt:self.numOutputs forKey:@"numOutputs"];
    
    [encoder encodeInt:self.maxIterations forKey:@"maxIterations"];
    [encoder encodeDouble:self.minError forKey:@"minError"];
    [encoder encodeDouble:self.learningRate forKey:@"learningRate"];
    [encoder encodeDouble:self.momentum forKey:@"momentum"];
    
    [encoder encodeBool:self.isTrained forKey:@"isTrained"];
    [encoder encodeObject:layersArray forKey:@"layers"];
}

@end
