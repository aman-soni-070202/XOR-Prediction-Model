import math, json, random

random.seed(42)

#     Then average all the squared errors → gives the final loss
def loss_function(pred, truth, algo):
    if algo == 'mse':
        return (pred - truth) ** 2
    elif algo == "bce":
        pred = min(max(pred, 1e-7), 1 - 1e-7)
        return -(truth*math.log(pred) + (1 - truth)*math.log(1 - pred))

def loss_derivation(pred, truth, algo):
    if algo == 'mse':
        return 2 * (pred - truth) #(because in mse  loss = (pred - true)^2, therefore, d(loss) = 2 * (pred - true)   (in derivation power get minus by 1 and the response get multiplied by original power value) )
    elif algo == "bce":
        return pred - truth


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


# Updating the weights and biases using the derivatives (calculated above) and a learning rate (set by us)
def updated_parameter(param, rate_of_loss, learning_rate):
    return param - learning_rate * rate_of_loss


# input_layer (only holds raw values)
data = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]


def init_weights(n_inputs):
    limit = 1 / math.sqrt(n_inputs)
    return [random.uniform(-limit, limit) for _ in range(n_inputs)]


def init_bias():
    return random.uniform(-0.1, 0.1)  # small bias to start

# Hidden Layer - 1 (2 inputs)
w1 = init_weights(2)
b1 = init_bias()

w2 = init_weights(2)
b2 = init_bias()

# Hidden Layer - 2 (2 neurons from hidden layer 1)
wh21 = init_weights(2)
bh21 = init_bias()

wh22 = init_weights(2)
bh22 = init_bias()

# Output layer (2 neurons from hidden layer 2)
w3 = init_weights(2)
b3 = init_bias()


print("Initial weights and biases:")
print("w1:", w1, "b1:", b1)
print("w2:", w2, "b2:", b2)
print("wh21:", wh21, "bh21:", bh21)
print("wh22:", wh22, "bh22:", bh22)
print("w3:", w3, "b3:", b3)


learning_rate = 0.5
epochs = 1500001
for epoch in range(1, epochs):
    total_loss = 0
    for i1, i2, o in data:
        # Forward Pass

        # Hidden layer 1 - neuron 1
        w_i_1 = generate_weighted_input([i1, i2], w1, b1)
        pred1 = activation_function(w_i_1, 'tanh')

        # Hidden layer 1 - neuron 2
        w_i_2 = generate_weighted_input([i1, i2], w2, b2)
        pred2 = activation_function(w_i_2, 'tanh')

        # Hidden layer 2 - neuron 1
        w_i_21 = generate_weighted_input([pred1, pred2], wh21, bh21)
        pred3 = activation_function(w_i_21, 'tanh')

        # Hidden layer 2 - neuron 2
        w_i_22 = generate_weighted_input([pred1, pred2], wh22, bh22)
        pred4 = activation_function(w_i_22, 'tanh')

        # Output layer
        w_i_3 = generate_weighted_input([pred3, pred4], w3, b3)
        final_pred = activation_function(w_i_3, 'sigmoid')

        # Calculate loss
        loss = loss_function(final_pred, o, 'bce')
        total_loss += loss

        if epoch % 1000 == 0:
            print(f"w_i_1: {w_i_1}, pred1: {pred1}")
            print(f"w_i_2: {w_i_2}, pred2: {pred2}")
            print(f"Input: ({i1}, {i2}) => Hidden 1: [{round(pred1, 3)}, {round(pred2, 3)}] => Hidden 2: [{round(pred3, 3)}, {round(pred4, 3)}] => Output: {round(final_pred, 3)} (Expected: {o}) => Loss: {round(loss, 5)}")

        # Backpropagation


        ## Calculating the rate of change for weight of output neuron
        derived_loss_wrt_final_pred = loss_derivation(final_pred, o, "bce") # gradient of loss with respect to the prediction
        derived_final_pred_wrt_w_i_3 = activated_derivative(w_i_3, 'sigmoid') # (we basically need the sigmoid of the wi3 which is final_pred)gradient of prediction with respect to the weighted input
        derived_loss_wrt_w_i_3 = derived_loss_wrt_final_pred * derived_final_pred_wrt_w_i_3

        ## For Output Neuron

        ### Gradients for output neuron weights and bias (chain rule)
        dL_dw3_0 = derived_loss_wrt_w_i_3 * pred3  # dL/dw3_0
        dL_dw3_1 = derived_loss_wrt_w_i_3 * pred4  # dL/dw3_1
        dL_db3 = derived_loss_wrt_w_i_3 * 1        # dL/db3

        ### Update output layer weights and bias
        w3[0] = updated_parameter(w3[0], dL_dw3_0, learning_rate)
        w3[1] = updated_parameter(w3[1], dL_dw3_1, learning_rate)
        b3 = updated_parameter(b3, dL_db3, learning_rate)

        ## For Hidden Neurons

        # Hidden Layer - 2

        ### Derivatives for hidden layer neurons (chain rule)
        derived_loss_wrt_pred3 = derived_loss_wrt_w_i_3 * w3[0]
        derived_pred3_wrt_w_i_21 = activated_derivative(w_i_21, 'tanh')

        derived_loss_wrt_pred4 = derived_loss_wrt_w_i_3 * w3[1]
        derived_pred4_wrt_w_i_22 = activated_derivative(w_i_22, 'tanh')

        dL_dw21_0 = derived_loss_wrt_pred3 * derived_pred3_wrt_w_i_21 * pred1
        dL_dw21_1 = derived_loss_wrt_pred3 * derived_pred3_wrt_w_i_21 * pred2
        dL_db21 = derived_loss_wrt_pred3 * derived_pred3_wrt_w_i_21

        dL_dw22_0 = derived_loss_wrt_pred4 * derived_pred4_wrt_w_i_22 * pred1
        dL_dw22_1 = derived_loss_wrt_pred4 * derived_pred4_wrt_w_i_22 * pred2
        dL_db22 = derived_loss_wrt_pred4 * derived_pred4_wrt_w_i_22

        ### Update hidden layer weights and biases
        wh21[0] = updated_parameter(wh21[0], dL_dw21_0, learning_rate)
        wh21[1] = updated_parameter(wh21[1], dL_dw21_1, learning_rate)
        bh21 = updated_parameter(bh21, dL_db21, learning_rate)

        wh22[0] = updated_parameter(wh22[0], dL_dw22_0, learning_rate)
        wh22[1] = updated_parameter(wh22[1], dL_dw22_1, learning_rate)
        bh22 = updated_parameter(bh22, dL_db22, learning_rate)

        # Hidden Layer - 1

        ### Derivatives for hidden layer neurons (chain rule)
        derived_loss_wrt_pred1 = (
            (derived_loss_wrt_pred3 * derived_pred3_wrt_w_i_21 * wh21[0]) +
            (derived_loss_wrt_pred4 * derived_pred4_wrt_w_i_22 * wh22[0])
        )
        derived_pred1_wrt_w_i_1 = activated_derivative(w_i_1, 'tanh')

        # derived_loss_wrt_pred2 = derived_loss_wrt_w_i_3 * w3[1]
        derived_loss_wrt_pred2 = (
            (derived_loss_wrt_pred3 * derived_pred3_wrt_w_i_21 * wh21[1]) +
            (derived_loss_wrt_pred4 * derived_pred4_wrt_w_i_22 * wh22[1])
        )
        derived_pred2_wrt_w_i_2 = activated_derivative(w_i_2, 'tanh')

        dL_dw1_0 = derived_loss_wrt_pred1 * derived_pred1_wrt_w_i_1 * i1
        dL_dw1_1 = derived_loss_wrt_pred1 * derived_pred1_wrt_w_i_1 * i2
        dL_db1 = derived_loss_wrt_pred1 * derived_pred1_wrt_w_i_1

        dL_dw2_0 = derived_loss_wrt_pred2 * derived_pred2_wrt_w_i_2 * i1
        dL_dw2_1 = derived_loss_wrt_pred2 * derived_pred2_wrt_w_i_2 * i2
        dL_db2 = derived_loss_wrt_pred2 * derived_pred2_wrt_w_i_2

        ### Update hidden layer weights and biases
        w1[0] = updated_parameter(w1[0], dL_dw1_0, learning_rate)
        w1[1] = updated_parameter(w1[1], dL_dw1_1, learning_rate)
        b1 = updated_parameter(b1, dL_db1, learning_rate)

        w2[0] = updated_parameter(w2[0], dL_dw2_0, learning_rate)
        w2[1] = updated_parameter(w2[1], dL_dw2_1, learning_rate)
        b2 = updated_parameter(b2, dL_db2, learning_rate)


    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Total Loss: {round(total_loss, 5)}")
        # print(f"Input: ({i1}, {i2}) => Hidden 1: [{round(pred1, 3)}, {round(pred2, 3)}] => Hidden 2: [{round(pred3, 3)}, {round(pred4, 3)}] => Output: {round(final_pred, 3)} (Expected: {o}) => Loss: {round(loss, 5)}")
        print('\n')

trained_data = {
    "hidden_layer": [
        [{"weights": w1, "bias": b1}, {"weights": w2, "bias": b2}],
        [{"weights": wh21, "bias": bh21}, {"weights": wh22, "bias": bh22}],
    ],
    "output_layer": {"weights": w3, "bias": b3}
}

json_object = json.dumps(trained_data, indent=4)

# Writing to sample.json
with open("weights_and_bias_bce_sigmoid.json", "w") as outfile:
    outfile.write(json_object)