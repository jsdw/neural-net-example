#!/usr/bin/env node

const Network = require("./network");

// Let's define our network. We follow the example network at
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example
// to follow along and see that everything works as expected.
//
// This network has an 2 input nodes, 2 hidden nodes, and 2 output nodes.
// each node is defined in terms of the weights connecting it to the previous
// layers nodes, and the bias weight connecting it to the bias node.
//
// As an example, the first hidden node is connected from the 2 input nodes
// with weights 0.15 and 0.2, and from a bias node with weight 0.35.
const network = new Network({
	network: [
		// input layer:
		[
			Network.node([],0),
			Network.node([],0)
		],
		// hidden layer:
		[
			Network.node([0.15, 0.2], 0.35),
			Network.node([0.25, 0.3], 0.35)
		],
		// output layer:
		[
			Network.node([0.4, 0.45], 0.6),
			Network.node([0.5, 0.55], 0.6)
		]
	],

	// the amount to multiply each proposed update by.
	updateMultiplier: 0.5,

	// The activation function takes some input value given to a node, and squashes
	// it. For back propagation, we'll also need to know what the derivative is of
	// this so that we can apply the chain rule and go backwards working out how
	// some weight affects the error.
	//
	// We use 1 / (1 + e^(-val)) as our example here (the logistic function).
	activationFunction: function(val) {
		return 1 / ( 1 + Math.exp(-val) );
	},

	// The derivative of the logistic function f(x) works out neatly to be
	// f(x) * (1 - f(x)):
	activationFunctionDerivative: function(val) {
		const a = 1 / ( 1 + Math.exp(-val) );
		return a * (1 - a);
	},

	// The error function compares some actual outputs with some expected ones,
	// to obtain an error. We need to be able to differentiate this and so we
	// also provide the derivative function for it to differentiate with respect
	// to a given actual output. To produce a total error, we will sum together
	// the errors obtained for each output value.
	//
	// 0.5*(expected - output)^2
	errorFunction: function(output, expected) {
		return 0.5 * Math.pow(expected - output, 2);
	},

	// the derivative of the error function with respect to some output. Since the total
	// error is the sum of errors for each output, all but the one error we care about
	// becomes constant with respect to a particular output, and so
	//
	// d(totalError)/d(someOutputN) = d(outputNError)/d(someOutputN)
	//
	// err = 0.5*(expected - output)^2 can be broken down into:
	// err = 0.5 * x^2
	// x = expected - output
	//
	// so chain rule can be used:
	// d(err)/d(output) = d(err)/d(x) * d(x)/d(output)
	//
	// which becomes:
	// (expected - output) * -1 = -(expected - output) = output - expected
	//
	errorFunctionDerivative: function(output, expected) {
		return output - expected;
	}
});

//
// We can run the network and print the output and error at each stage to verify that
// it works as expected:
//

function printStats(network, step) {
	console.log(`========== T = ${step} ==========`);
	console.log(`Output:`, network.outputs());
	console.log(`The error is: ${network.totalError([0.01, 0.99])}`);
}

network.feedInputs([0.05, 0.10]);
printStats(network, 0);

for(let i = 1; i <= 10000; i++) {
	network.trainingStep([0.05, 0.10], [0.01, 0.99]);
	if(i === 1 || i % 1000 == 0) printStats(network, i);
}
