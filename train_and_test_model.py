import model

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # number of units in input layer- Basically how many features you have
n_h = 7         # number of hidden layers you want
n_y = 1         # number of output units.
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


###Data

'''
A training set of  images labelled as cat (1) or non-cat (0)
A test set of `m_test` images labelled as cat and non-cat
Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
'''
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255
test_x = test_x_flatten/255
#Note:  12,288  equals  64×64×3 , which is the size of one reshaped image vector.


###Train the model
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


###Test with your own image
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(image, my_label_y, parameters)


print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")