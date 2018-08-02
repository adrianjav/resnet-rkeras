# ResNet in Keras for R

This is the simplest implementation of ResNet in Keras for R you can think of. It's quite short and limited by now, but I'll try to add more features in the future. It's also missing some auxiliary functions I was using to plot confidence intervals and so on, I'll upload a Jupyter notebook any time soon.

The implementation is based on [this one](https://github.com/facebook/fb.resnet.torch/) written in Lua with the Torch Framework. It also implements the small tweak of removing the ReLU activations at the end of each residual block as described [here](http://torch.ch/blog/2016/02/04/resnets.html).

A simple example of how to use the code is shown below.

```r
source('resnet.R')

# Taking a subset of the Cifar-10 dataset
cifar10 <- dataset_cifar10()
cifar10.orig <- cifar10

x_train <- cifar10$train$x[1:100,,,]
y_train <- cifar10$train$y[1:100,]
x_test <- cifar10$test$x[1:10,,,]
y_test <- cifar10$test$y[1:10,]

y_tags <- y_train
y_train <- to_categorical(y_train)
y_test <- to_categorical(y_test)

model <- build_resnet_cifar10(20)

# Doing cross validation (it concatenates all the results)
model.cv <- do.cross.validation.resnet(20,
	    x_train, y_train, batch_size=5, 
	    epochs=10, y_tags=y_tags, k=5,
	    loss='categorical_crossentropy', 
	    metrics=c('accuracy') 
    )

# Compiling and training the model
model %>%
	compile(
  	  optimizer=optimizer_sgd(lr=0.1, momentum=0.9, decay=0.0001),
  	  loss='categorical_crossentropy', metrics=c('accuracy')
  	  ) %>%
	fit(
    	x_train, y_train, validation_split=0.2,
    	verbose=0, batch_size=5, epochs=10,
    	callbacks = c(callback_reduce_lr_on_plateau(verbose=0, patience=10, factor=0.1))
    )
 
 # Getting and plotting the predictions
 predictions <- predict(model, x_test)
 print(paste('Predictions:', paste0(max.col(predictions), collapse=' ')))
 print(paste('Predictions:', paste0(max.col(y_test), collapse=' ')))
```