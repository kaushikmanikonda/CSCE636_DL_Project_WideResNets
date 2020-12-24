# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

preprocess_configs = {
	"crop" : True,
	"crop_padding" : 4,
	"flip" : True,
 	"cutout" : True,
	"cutout_holes": 1,
	"cutout_length": 16,
	"batch_size": 128,

	# ...
}

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 28,
	"num_classes": 10,
	"width_multiplier": 10,
	"dropRate": 0.3
}

training_configs = {
	"batch_size": 128,
	"learning_rate": 0.1,
	"epochs": 251,

	# ...
}

### END CODE HERE