// The entry point for doing networky things. it expects to be passed a network
// definition that looks something like that seen in the included example.js.
function Network(definition) {
	this.definition = definition;
}

// Create nodes using this function, which also prepares each node to
// store cached values needed when training.
Network.node = function(weights, bias) {
	return {
		weights: weights,
		bias: bias,
		cached: {
			// the input seen total input to this node:
			input: 0,
			// the last seen total output from this node activationFunction(input):
			output: 0,
			// the proposed updates for each weight (same length as weights):
			weightUpdates: weights.slice().map(_ => 0),
			// the proposed update for the bias weight:
			biasUpdate: 0,
			// during backprop, we cache d(input)/d(totalError) since we'll
			// want to make use of that to differentiate previous layer weights wrt error:
			errorWrtIn: 0
		}
	}
}

// doing a forward pass basically involves setting input nodes to be
// outputting those values, and then for each node multiplying the
// inputs to the node by the weights of those connections, remembering
// to also add the bias input * weight (1 * weight = weight).
Network.prototype.feedInputs = function(inputs){

	const network = this.definition.network;
	const activationFunction = this.definition.activationFunction;

	// set the input nodes to be firing the input values provided:
	network[0].forEach(function(node, i){
		node.cached.output = inputs[i];
	});

	// for each non-input layer, forward propagate the values:
	for(let i = 1; i < network.length; i++) {

		const lastLayer = network[i-1];
		const layer = network[i];

		layer.forEach(node => {

			// The input to a ndoe is the output from each node on the previous layer multiplied
			// by the weight of the connection between them:
			let input = node.weights.reduce((v, w, i) => v + lastLayer[i].cached.output * w, 0);

			// add the bias value:
			input += node.bias;

			// cache the input to this node and the output after applying the activationFunction:
			node.cached.input = input;
			node.cached.output = activationFunction(input);;

		});

	}

};

// give the current output values for the current state of the network:
Network.prototype.outputs = function(){

	const network = this.definition.network;

	return network[network.length-1].map(function(node){
		return node.cached.output;
	});

};

// This returns the error between some expected outputs and
// the current network outputs.
Network.prototype.totalError = function(expected) {

	const network = this.definition.network;
	const errorFunction = this.definition.errorFunction;

	return network[network.length-1].reduce((sum, node, i) => {
		return sum + errorFunction(node.cached.output, expected[i]);
	}, 0);

};

// Perform a training step using back propagation on the
// network given its current state.
Network.prototype.trainingStep = function(inputs, expected) {

	const network = this.definition.network;
	const errorFunctionDerivative = this.definition.errorFunctionDerivative;
	const activationFunctionDerivative = this.definition.activationFunctionDerivative;
	const updateMultiplier = this.definition.updateMultiplier;

	this.feedInputs(inputs);

	const error = this.totalError(expected);
	const outputs = this.outputs();

	// first, propose updates for each output node weight.
	const previousLayer = network[network.length-2];
	network[network.length-1].forEach((node, i) => {

		// how much does each weight connected to output affect the error?
		//
		// In effect, we have taken the giant function that the neural network represents (which goes all
		// the way from some inputs values to some error value), and looks a little like:
		//
		// error = 0.5*(expected - out1)^2 + ... + 1/2(expected - outN)^2
		// out1 = activationFunction(in1)
		// in1 = w1*prevOut1 + w2*prevOut2 + ... + wN*prevOutN + bias*1
		//
		// and we want to calculate the change in error with respect to some weight, to see in which
		// direction the weight needs updating in order to reduce the error. So, we want to work out,
		// in the case of updating w1:
		//
		// d(error)/d(w1)
		//
		// And given the chain rule, this means we'll have to do sometihng like:
		//
		// d(error)/d(w1) = d(error)/d(out1) * d(out1)/d(in1) * d(in1)/d(w1)
		//
		// where:
		// - d(error)/d(out1) is the derivative of the error function with respect to the one node output
		// value we care about. This basically means that all but one 0.5*(expected-out1)^2 are constant with
		// respect to out1 changing, and disappear.
		// - d(out1)/d(in1) basically involves calculating the derivative of the activation function, since that's
		// what takes us from in1 to out1 (remember, out1 = activationFunction(in1)).
		// - d(in1)/d(w1) digs into the giant function that results in some in1, and differentiates
		// it with respect to a weight we care about updating. in1 looks like w1*prevOut1 + ... + wN*prevOutN + bias*1,
		// and d(w1*prevOut1 + ... + wN*prevOutN + bias*1)/d(w1) = d(w1*prevOut1)/d(w1) = prevOut1, since w.r.t
		// w1, the only part that changes as w1 does is w1*prevOut1.
		//

		// find d(error)/d(out)
		const output = node.cached.output;
		const errorWrtOut = errorFunctionDerivative(output, expected[i]);

		// find d(out)/d(in)
		const input = node.cached.input;
		const outWrtIn = activationFunctionDerivative(input);

		// chain these to find d(error)/(in)
		const errorWrtIn = errorWrtOut * outWrtIn;

		// cache errorWrtIn since we'll want it for prev layer calculations:
		node.cached.errorWrtIn = errorWrtIn;

		// proposed weight updates for this node:
		node.cached.weightUpdates = node.weights.map((weight, i) => {

			// propose an update each weight by finding d(in)/d(weight), applying
			// the chain rule to find d(error)/d(weight), and then multiplying the
			// result by our updateMultiplier:
			const inWrtWeight = previousLayer[i].cached.output;

			// remember, that we want to move in the opposite direction to the gradient,
			// so we add a minus sign.
			return -(errorWrtIn * inWrtWeight * updateMultiplier);

		});

		// the bias node always outputs 1, which = inWrtWeight, so nice and easy:
		node.cached.biasUpdate = errorWrtIn * 1 * updateMultiplier;

	});

	// next, go back through the layers proposing updates for hidden node weights.
	// we go backwards because a bunch of computation we'll need at each layer has
	// already been done for us in the layer+1 update step.
	for(let i = network.length-2; i > 0; i--) {

		const previousLayer = network[i-1];
		const nextLayer = network[i+1];
		network[i].forEach((node, i) => {

			// so, we still just need to find d(error)/d(w) for each weight w in this
			// layer. For output node weights, only one output node value (and so error)
			// changes w.r.t the weight, because all others remain constant as the weight
			// changes and so have no effect on the final result (and so disappear).
			//
			// For hidden layers, every node in the next layer matters, since all of their
			// values change w.r.t a change in some hidden layer weight.
			//
			// ultimately, for each node in the next layer we have already worked out
			// d(error)/d(next_in) for that node. For each of those, we want to get to
			// d(error)/d(w). by chaining, this means doing:
			//
			// d(error)/d(next_in) * d(next_in)/d(out) * d(out)/d(in) * d(in)/d(w)
			//
			// Since those values are then summed, the differential
			// of them is as well, so we need SUM( d(error)/d(w) ) for each next node.
			//

			// d(error)/d(out), summed for each node because the equation that matters
			// at this level is a sum of things, all of which matter w.r.t out.
			const errorWrtOut = nextLayer.reduce((sum,nextNode) => {

				// we have already worked this out, so use the same value:
				const errorWrtNextInput = nextNode.cached.errorWrtIn;

				// remember that nextInput = out1*w1 + ... + outN*wN, and
				// d(out1*w1 + ... + outN*wN)/d(out1) differentiates to w1:
				const nextInputWrtOut = nextNode.weights[i];

				// calculate d(error)/d(out) and sum for each node.
				return sum + errorWrtNextInput * nextInputWrtOut;

			},0);

			// d(out)/d(in), remembering that out = activationFunction(in).
			// this is the same for every node in next layer, so we only need
			// to calculate it once.
			const outWrtIn = activationFunctionDerivative(node.cached.input);

			// chain the prev two calculations to get d(error)/d(in):
			const errorWrtIn = errorWrtOut * outWrtIn;

			// cache to use in the calculation for the next layer down:
			node.cached.errorWrtIn = errorWrtIn;

			// now, weight updates are the same as for the output layer:
			node.cached.weightUpdates = node.weights.map((weight, i) => {

				const inWrtWeight = previousLayer[i].cached.output;
				return -1 * errorWrtIn * inWrtWeight * updateMultiplier;

			});

			// .. as are bias updates:
			node.cached.biasUpdate = errorWrtIn * 1 * updateMultiplier;

		});

    }

	// now that we've found all of the proposed updates, we can apply them to the network,
	// ending our training step.
	network.forEach(layer => {
		layer.forEach(node => {
			node.weights = node.weights.map((w,i) => w + node.cached.weightUpdates[i])
		});
	});

};

module.exports = Network;