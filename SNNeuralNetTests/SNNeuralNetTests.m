//
//  SNNeuralNetTests.m
//  SNNeuralNetTests
//
//  Created by Devon Govett on 12/1/13.
//  Copyright (c) 2013 Devon Govett. All rights reserved.
//

#import <XCTest/XCTest.h>
#import "SNNeuralNet.h"

@interface SNNeuralNetTests : XCTestCase

@end

@implementation SNNeuralNetTests

- (void)_testBitwise:(SNTrainingRecord *)records
          numRecords:(int)numRecords
           numInputs:(int)numInputs
{
    SNNeuralNet *net = [[SNNeuralNet alloc] initWithInputs:numInputs outputs:1];
    [net train:records numRecords:numRecords];
    
    XCTAssertEqual(net.isTrained, YES, @"");
    
    for (int i = 0; i < numRecords; i++) {
        double *output = [net runInput:records[i].input];
        XCTAssertEqualWithAccuracy(*output, records[i].output[0], 0.15, @"");
    }
}

- (void)testNOT
{
    SNTrainingRecord records[] = {
        {SNInput(0), SNOutput(1)},
        {SNInput(1), SNOutput(0)}
    };
    
    [self _testBitwise:records numRecords:2 numInputs:2];
}

- (void)testXOR
{
    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(1)},
        {SNInput(1,0), SNOutput(1)},
        {SNInput(1,1), SNOutput(0)}
    };
    
    [self _testBitwise:records numRecords:4 numInputs:2];
}

- (void)testOR
{
    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(1)},
        {SNInput(1,0), SNOutput(1)},
        {SNInput(1,1), SNOutput(1)}
    };
    
    [self _testBitwise:records numRecords:4 numInputs:2];
}

- (void)testAND
{
    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(0)},
        {SNInput(1,0), SNOutput(0)},
        {SNInput(1,1), SNOutput(1)}
    };
    
    [self _testBitwise:records numRecords:4 numInputs:2];
}

- (void)testColors
{
    // inputs are rgb colors
    // outputs are booleans [white, black] for text color
    SNTrainingRecord records[] = {
        {SNInput(0.03, 0.7, 0.5), SNOutput(0, 1)},
        {SNInput(0.16, 0.09, 0.2), SNOutput(1, 0)},
        {SNInput(0.5, 0.5, 1.0), SNOutput(1, 0)}
    };
    
    SNNeuralNet *net = [[SNNeuralNet alloc] initWithTrainingData:records numRecords:3 numInputs:3 numOutputs:2];
    double *output = [net runInput:SNInput(1, 0.4, 0)];
    
    XCTAssertEqualWithAccuracy(output[0], 0.984, 0.05, @"");
    XCTAssertEqualWithAccuracy(output[1], 0.015, 0.05, @"");
}

- (void)testMinError
{
    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(0)},
        {SNInput(1,0), SNOutput(0)},
        {SNInput(1,1), SNOutput(1)}
    };
    
    SNNeuralNet *net = [[SNNeuralNet alloc] initWithInputs:2 outputs:1];
    net.minError = 0.2;
    double error = [net train:records numRecords:4];
    
    XCTAssert(error < 0.2, @"network did not train until error threshold was reached");
}

- (void)testHiddenLayers
{
    SNTrainingRecord records[] = {
        {SNInput(0,0), SNOutput(0)},
        {SNInput(0,1), SNOutput(0)},
        {SNInput(1,0), SNOutput(0)},
        {SNInput(1,1), SNOutput(1)}
    };
    
    SNNeuralNet *net = [[SNNeuralNet alloc] initWithInputs:2 hiddenLayers:@[@3, @4] outputs:1];
    net.minError = 0.2;
    double error = [net train:records numRecords:4];
    
    XCTAssert(error < 0.2, @"network did not train until error threshold was reached");
}

@end
