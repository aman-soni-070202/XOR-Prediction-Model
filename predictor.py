import math, json

# Activation Function ==> to flatten the input in range of 0 to 1
def activation_function(x, algo):
    if algo == 'sigmoid':
        return 1 / (1 + math.exp(-x))
    elif algo == 'tanh':
        return math.tanh(x)


# Derivative of sigmoid (needed for training) (tells us how much sigmoid output(prediction) change when z(weighted input) changes  - which then tells how much we have to change the weight to reduce the error score)
def activated_derivative(x, algo):
    if algo == 'sigmoid':
        sx = activation_function(x, algo)
        return sx * (1 - sx)
    elif algo == 'tanh':
        tx = activation_function(x, algo)
        return 1 - tx ** 2


# to generate the weighted input by adding the weights and biases
def generate_weighted_input(inputs: list, weights: list, bias: int):
    weighted_input = 0

    for i in range(len(inputs)):
        weighted_input += (inputs[i]*weights[i])

    weighted_input += bias

    return weighted_input


def predict(i1: int, i2: int, weights: dict, biases: dict, activation_algo: str):
    w_i_1 = generate_weighted_input([i1, i2], weights['h1_n1'], biases['h1_n1'])
    pred1 = activation_function(w_i_1, activation_algo)

    # Hidden layer 1 - neuron 2
    w_i_2 = generate_weighted_input([i1, i2], weights['h1_n2'], biases['h1_n2'])
    pred2 = activation_function(w_i_2, activation_algo)

    # Hidden layer 2 - neuron 1
    w_i_21 = generate_weighted_input([pred1, pred2], weights['h2_n1'], biases['h2_n1'])
    pred3 = activation_function(w_i_21, activation_algo)

    # Hidden layer 2 - neuron 2
    w_i_22 = generate_weighted_input([pred1, pred2], weights['h2_n2'], biases['h2_n2'])
    pred4 = activation_function(w_i_22, activation_algo)

    # Output layer
    w_i_3 = generate_weighted_input([pred3, pred4], weights['ol'], biases['ol'])
    final_pred = activation_function(w_i_3, activation_algo)

    print(round(final_pred))


loss_algo = "mse"
activation_algo = "tanh"

weights_and_bias = {}
with open(f'weights_and_bias_{loss_algo}_{activation_algo}.json', 'r') as openfile:
    weights_and_bias = json.load(openfile)

if weights_and_bias:
    weights = {
        f"h{i+1}_n{j+1}": weights_and_bias['hidden_layer'][i][j]['weights'] 
        for i in range(len(weights_and_bias['hidden_layer']))
        for j in range(len(weights_and_bias['hidden_layer'][i]))
    }
    weights['ol'] = weights_and_bias['output_layer']['weights']

    biases = {
        f"h{i+1}_n{j+1}": weights_and_bias['hidden_layer'][i][j]['bias'] 
        for i in range(len(weights_and_bias['hidden_layer']))
        for j in range(len(weights_and_bias['hidden_layer'][i]))
    }
    biases['ol'] = weights_and_bias['output_layer']['bias']

    predict(1, 1, weights, biases, activation_algo)