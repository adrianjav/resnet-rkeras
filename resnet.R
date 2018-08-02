library(keras)
library(caret)
library(dplyr) # For the %>% operator

# Functions to build the model

layer_shortcut <- function(object, ninput, noutput, stride) {
	if(ninput == noutput)
		object %>% 
			layer_lambda(function(x) x)
	else{
		object <- object %>%
			layer_average_pooling_2d(1, stride)

		a <- object %>%
			layer_lambda(function(x) x)

		b <- object %>% 
			layer_lambda(., function(x) k_zeros_like(x))

		layer_concatenate(c(a, b))
	}
}

layer_basic_block <- function(object, ninput, noutput, stride) {
	a <- object %>%	
		layer_conv_2d(noutput, 3, stride, 'same', kernel_initializer = 'lecun_normal') %>%
		layer_batch_normalization() %>%
		layer_activation('relu') %>%
		layer_conv_2d(noutput, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
		layer_batch_normalization()

	b <- object %>%
		layer_shortcut(ninput, noutput, stride)

	layer_add(c(a, b)) #%>%
		#layer_activation('relu') QUITAMOS ESTO
}


build_block <- function(object, ninput, noutput, count, stride) {
	for(i in 1:count)
		object <- object %>% 
			layer_basic_block(if(i == 1) ninput else noutput,
			                  noutput, 
			                  if(i == 1) stride else 1
			                  )
	object
}

build_resnet_cifar10 <- function(depth = 20) { 
	n <- (depth - 2) / 6

	input <- layer_input(shape=c(32,32,3))
	
	output <- input %>%
		layer_conv_2d(16, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
		layer_batch_normalization() %>%
		layer_activation('relu') %>%
		build_block(16, 16, n, 1) %>% # Primer conjunto
		build_block(16, 32, n, 2) %>% # Segundo conjunto
		build_block(32, 64, n, 2) %>% # Tercer conjunto
		layer_average_pooling_2d(8, 1) %>%
		layer_flatten() %>%
		layer_dense(10) %>%
		layer_activation_softmax()		

	keras_model(input, output)
}

# Functions to perform a stratified cross-validation

do.cross.validation.resnet <- function(depth, x, y, batch_size, epochs, 
                                y_tags=NULL, k = 5, callbacks = NULL, ...) {
  folds <- createFolds(y = if(is.null(y_tags)) y else y_tags, 
                       k = k, list = F) # Stratified
  histories <- list()
  for(f in 1:k){
    print(paste(f, 'of', k))
    
  	model.aux <- build_resnet_cifar10(depth) %>% compile(
  	  optimizer=optimizer_sgd(lr=0.1, momentum=0.9, decay=0.0001), 
  	  ...
  	  ) 
    ind <- which(folds == f)
    x_train <- x[-ind,,,]
    y_train <- y[-ind,]
    x_valid <- x[ind,,,]
    y_valid <- y[ind,]
    
    histories[[f]] <- model.aux %>% fit(
      x_train, y_train,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(x_valid, y_valid),
      verbose = 0,
      callbacks = c(callback_reduce_lr_on_plateau(verbose=0, patience=10, factor=0.1))
    )
  }
  histories
}

