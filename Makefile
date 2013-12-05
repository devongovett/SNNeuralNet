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
	
docs:
	rm -rf docs
	appledoc --project-name SNNeuralNet \
					 --project-company "Devon Govett" \
					 --company-id com.devongovett.SNNeuralNet \
					 --keep-intermediate-files \
					 --no-create-docset \
					 --create-html \
					 --ignore *.m \
					 --no-warn-missing-output-path \
					 --exit-threshold 2 \
					 SNNeuralNet/
	mv html docs

.PHONY: clean build test docs
.DEFAULT_GOAL := test
