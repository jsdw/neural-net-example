// a basic NN implementation based on the example provided at:
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example

// First, let's define our network.
//
// each node (in weights) connects to every node on the next layer.
// there is one bias node per layer, whose weight is updated in the
// same way as every other node, but whose value is always 1 and is not
// tied to any values of nodes on other layers.
//
// This network has an 2 input nodes, 2 hidden nodes, and 2 output nodes.
// Here, input node 1 (i1) has weights to hidden node 1 and hidden node 2
// (h2 and h2) or 0.15 and 0.2. Input node 2 (i2) has weights to h1 and h2
// of 0.25 and 0.3.
const network = [

	{
		weights: [[0.15, 0.2], [0.25, 0.3]],
		bias: 0.35
	},

	{
		weights: [[0.40, 0.45], [0.5, 0.55]],
		bias: 0.6
	}

];

// The activation function takes some input value given to a node, and squashes
// it. For back propagation, we'll also need to know what the derivative is of
// this so that we can apply the chain rule and go backwards working out how
// some weight affects the error.
//
// We use 1 / (1 + e^val) as our example here (the logistic function).
function activationFunction(val) {
	return 1 / ( 1 + Math.exp(val) );
}

// The derivative of the logistic function f(x) works out neatly to be
// f(x) * (1 - f(x)):
function activationFunctionDerivative(val) {
	const a = activationFunction(val);
	return a * (1 - a);
}

// The error function compares some actual outputs with some expected ones,
// to obtain an error. We need to be able to differentiate this and so we
// also provide the derivative function for it to differentiate with respect
// to a given actual output.
//
// This is SUM( 0.5*(expectedOutputs[i] - actualOutputs[i])^2 ).
function errorFunction(actualOutputs, expectedOutputs) {
	let output = 0;
	actualOutputs.forEach(function(actual, i){
		const expected = expectedOutputs[i];
		output += 0.5 * Math.pow(expected - actual, 2);
	});
	return output;
}

// the derivative of the error function with respect to some output (ie actualOutputs[i]).
//
// most of the function turns to constants and derives to 0, so we just need the derivative
// of 0.5*(expectedOutputs[i] - actualOutputs[i])^2. using chain rule it becomes
// (expectedOutputs[i] - actualOutputs[i]) * -1. (outside w.r.t inside, then inside
// w.r.t actualOutputs[i]).
function errorFunctionPartialDerivate(actualOutputs, expectedOutputs, i) {
	const actual = actualOutputs[i];
	const expected = expectedOutputs[i];
	return (actual - expected) * -1;
}

// run a network, giving back actualOutputs. The forward pass is very simple; For some node,
// just multiply all of the weights into it by the inputs fed to them, and then apply the
// activation function to arrive at an output. repeat for each layer until we have output
// values.
function forwardPass(network, inputs) {

	network.forEach(layer => {

		// for each node..
		inputs = layer.map(inputWeights => {
			// sum up input values * weights into this node:
			let val = inputWeights.reduce((v, w, inputIndex) => v + inputs[inputIndex] * w, 0);
			// apply the activation function to the resulting value:
			return activationFunction(val);
		});

	});
	return inputs;

}