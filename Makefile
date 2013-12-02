MAC_OPTS := -project SNNeuralNet.xcodeproj -scheme 'SNNeuralNet.framework'
IOS_OPTS := -project SNNeuralNet.xcodeproj -scheme 'SNNeuralNet'

clean:
	xctool $(IOS_OPTS) -sdk iphonesimulator -configuration Release clean
	xctool $(MAC_OPTS) -sdk macosx -configuration Release clean
	
build:
	xctool $(IOS_OPTS) -sdk iphonesimulator -configuration Release build
	xctool $(MAC_OPTS) -sdk macosx -configuration Release build
	
test: build
	xctool $(IOS_OPTS) -sdk iphonesimulator -configuration Release test -test-sdk iphonesimulator
	xctool $(MAC_OPTS) -sdk macosx -configuration Release test -test-sdk macosx

.PHONY: clean build test
.DEFAULT_GOAL := test
