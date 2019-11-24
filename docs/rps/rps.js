

//definitions
var moves = ["ROCK", "PAPER", "SCISSORS"];
var myMove;
var opMove;
var opMoveIdx;

/**
 *The most common type of model is the Sequential model, which is a linear stack of layers. 
 *You can create a Sequential model by passing a list of layers to the sequential() function
 *Here we associate the sequential function with the identifier "model" for this code
 */
const model = tf.sequential();

/**
 *Initialize the model parameters and plot the initial probabilities.
 */
function init(){
    /* Build sequential model layer by layer (just one here)
         tf.layers.dense   = ordinary dense layer of nodes
         units             = number of output nodes
         useBias           = is there a weight zero in the calculations? 
                             (see https://medium.com/deeper-learning/glossary-of-deep-learning-bias-cf49d9c895e2)
         kernelInitializer = kernel refers to the weights other than w0. In this case weights drawn from 
                             a truncated normal distribution centered on 0 with standarddeviation = sqrt(2 / fan_in)
                             where fan_in depends on the size of the previous layer of neurons.
    */
	model.add(tf.layers.dense({units: 3, inputShape: [3], useBias: false, kernelInitializer: 'heNormal'}));
    
    /* "compile" the model: specify a loss function and an optimizer
         loss             = how do we compute the total difference between predicted and expected value
         optimizer        = how will we move step by step toward better weights.  
                            Here sgd = Stochastic Gradient descent = instead of actual gradient 
                            (calculated from lots of data) it uses a statistical estimate of  
                            the gradient (less computational work)
    */
	model.compile({loss: 'meanSquaredError', optimizer: 'sgd'}); 
    
    //plotProbs is a function defined below
	plotProbabilities();
}

/*
 * Choose an agent's move based on the phase.
 * Training Phase: Computrer chooses actions randomly
 * Evaluation Phase: Choose actions based on argmax probability.
 *   argmax probability = compute f(x) for a bunch of x's and then choose the x that yields largest result.
 * Params: My Move
 * Return: Agent Move
 *
 * Triggered by user clicking on a move: setAgentMove(chooseMove(this.value))
 */

function chooseMove(move){
	
	myMove = move;
    //Are we in training phase or testing phase?
	var phase = document.getElementsByName('phase');

	if(phase[1].checked){ //in testing phase the agent will choose from model
        //convertToOneHot changes my choice (R, S, or P) to [1,0,0],  [0,1,0], or  [0,0,1]
		var intMove = convertToOneHot(myMove);
        //use tensorflow to convert [0,1,0], for example, into a rank-2, 3 element tensor 
        // the result will be [[0, 1, 0],]
		var xs = tf.tensor2d(intMove, [1,3]);
        //run the model on the x's and arraySync() returns tensor result as a nested array: number[]
        //we get three probabilities: one for R, one for S, one for P
		var logits = model.predict(xs).arraySync()[0];
		//getMaxIndex scans the probabilities and returns the index of the top one
        //that will be agent's move
		opMove = moves[getMaxIndex(logits)];
	}else{
		//choose randomly 
	 	opMoveIdx = Math.floor((Math.random() * 3) + 0);
		opMove = moves[opMoveIdx];
	}

	return opMove;
}

/**
 * Plot the probabilities of the choosing a move for each of the my moves.
 */

function plotProbabilities(){
	var divs = ['div1', 'div2', 'div3']
	var probs;
	var data;
	var xs;
	var logits;
	for(var i=0;i<3;i++){
        //generate tensor for move[i]
		xs = tf.tensor2d(convertToOneHot(moves[i]), [1, 3]);
        //run the model to get predictions
		logits = model.predict(xs).arraySync()[0];
        //convert the predictions into probabilities (3 element array)
		probs = tf.softmax(logits).arraySync();
		data = [
			{
				x:moves,
				y:probs,
				type:'bar'
			}
		];

		var layout = {
			title: 'What should I play against ' + moves[i] + '?',
			width: 450,
			height: 300

		};
		Plotly.newPlot(divs[i], data, layout);
	}
}

/**
 * Trains the model based on the reward given by the user. Triggered by button click.
 *
 * Params: reward
 * Return: None
 */
function train(reward){
	var phase = document.getElementsByName('phase')
	if(phase[0].checked){ //TRAINING MODE
		// convert selected move to 3 element vector with 1 at choice, 0s elsewhere
        // myMove is global variable set by chooseMove
		var intMove = convertToOneHot(myMove);
        // convert that array into a rank 2 tensor
		var xs = tf.tensor2d(intMove, [1,3]);
        // run this move through current network
        // logits are just the predictions for this set of x's
		var logits = model.predict(xs).arraySync()[0];

		//update model
        //during training opMoveIndex is set randomly
        console.log ('logits = ', logits)
        //pile reward on the agent's move choice
		logits[opMoveIdx] = logits[opMoveIdx] + reward;
        console.log ('logits = ', logits)
		const ys = tf.tensor2d(logits, [1, 3])
        //pass the x's and the y's to model to recompute weights
        // wait for it to finish and then 
        // fit returns: Promise<History>. Upon completion, hands no data to function plotProbs
		model.fit(xs, ys).then(()=>{
			plotProbabilities();
		});
	}
}

/**
 * Converts a move into a one-hot vector.
 *
 * Params: Move
 * Return: One-Hot-Vector
 */
function convertToOneHot(move){
	if(move=="ROCK") return [1, 0, 0];
	if(move=="PAPER") return [0, 1, 0];
	if(move=="SCISSORS") return [0, 0, 1];
	//throw error
}

/**
 * Choose the index of the maximum value from the array.
 *
 * Params: Array of values
 * Return: Index of the max value
 */
function getMaxIndex(values){
	var max=values[0];
	var index=0;

	for(var i=1;i<values.length;i++){
		if(values[i]>max){
			max = values[i];
			index = i;
		}
	}
	return index;
}