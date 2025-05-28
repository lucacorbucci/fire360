# FIRE360

A comprehensive and fast local explanation approach tailored for tabular data.

## How to use

You can find an example of how to use FIRE360 in the `example` folder. 

## Dataset Pre-Processing 


We evaluated FIRE360 on six datasets: Adult, Dutch, Covertype, House16, Letter, and Shuttle.
For each dataset, we performed the following pre-processing: 
* Adult: we encoded the categorical variables using OneHot encoding, and then we applied a MinMaxScaler.
* Dutch is a numeric dataset, therefore, we only used a MinMaxScaler.
* For Letter, since all the variables were numerical, we only used a MinMaxScaler.
* For Shuttle, since all the variables were numerical, we only used a MinMaxScaler.
* House16, since all the variables were numerical, we only used a MinMaxScaler.
* For Covertype, we converted the categorical variables using OneHot encoding, and then we applied a MinMaxScaler.

Training of the black-box models
\label{appendix:training_bb}

To simulate the entire explanation pipeline, we trained a black-box model for each dataset we used to evaluate FIRE360. In particular, we tune all hyperparameters for each dataset using Bayesian optimization implemented in Weights & Biases, maximizing model accuracy. The tuned hyperparameters are the following: batch size (we tested values in the range 16-64), learning rate (values between 0.0001 and 0.1) and optimizer (with ``Adam'' and ``SGD'' as possible values).


We used two different neural network architectures, one for the datasets with binary classification tasks (Adult, Dutch and House16) and one for the multiclass datasets (Letter, Covertype and Shuttle). In particular, for the binary case, we used a basic feedforward neural network with a single layer with 32 units followed by a Rectified Linear Unit (ReLU) activation function. For multiclass datasets, we used a feedforward neural network with 2 layers, the first one with 32 and the second with 64 hidden units. Both of them are followed by the ReLU activation function. 


Training the synthethisers

We trained the synthesizers using CTGAN and TVAE. In particular, following the suggestions of the developers of the [SDV library](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer\#how-do-i-tune-the-hyperparameters-such-as-epochs-or-other-values), we trained the synthesizers using three possible values for the number of epochs (1000, 2500 and 5000). We report here in the Appendix two plots of the training losses obtained when training the synthesizers on Adult and Letter. In particular, in Figure~\ref{fig:adult_ctgan} and Figure~\ref{fig:letter_ctgan}, we show the losses of the discriminator and of the generator obtained when using the Adult dataset. As you can see, increasing the number of epochs of the training of the synthesizers has no visible impact on the two losses. Therefore, we decided to use 2500 epochs to create our final synthetic dataset using the CTGAN. 
We report in Figure~\ref{fig:tvae_losses} a similar plot with the losses obtained when training the TVAE synthesizers with Adult and Letter. Even in this case, there is no big difference between the possible values of the epochs.

![Loss of the discriminator and of the generator obtained when training the CTGAN with the Adult Dataset. Increasing the number of epochs has no visible impact on the two losses](https://github.com/lucacorbucci/fire360/blob/main/images/gan_losses/adult_ctgan_losses.png?raw=true)

![Loss of the discriminator and of the generator obtained when training the CTGAN with the Letter Dataset. Increasing the number of epochs has no visible impact on the two losses.](https://github.com/lucacorbucci/fire360/blob/main/images/gan_losses/letter_ctgan_losses.png?raw=true)

![Adult tvae](https://github.com/lucacorbucci/fire360/blob/main/images/gan_losses/adult_tvae_loss.png?raw=true)

![Letter tvae](https://github.com/lucacorbucci/fire360/blob/main/images/gan_losses/letter_tvae_loss.png?raw=true)





Surrogate models grid search

Before training the surrogate white-box, we perform a simple grid search to ensure that it performs well. To perform the grid search we divided the dataset $\chi$ composed of samples taken from the synthetic dataset into two parts: train which represent the 80% of the dataset and test which represents the 20\%. For each grid search, we performed a KFold cross-validation with n_splits=5.
More specifically, in the grid search, we considered the following hyperparameters:
* Logistic Regression: We searched for two possible values of penalty (``l1'' and ``l2''), two values for the class\_weight (None and ``balanced'') and three values for the hyperparameter C (0.01, 0.1 and 1).
* SVM: We searched for 5 possible values of the parameter C (0.01, 0.1, 1, 10 and 100), two values of class\_weight (None and ``balanced'')
* KNN: We searched for 4 possible values of n\_neighbors (3, 5, 7, 9), two values of weights (``uniform'', ``distance'') and two values of metric (``Euclidean'', ``manhattan'')
* Decision Tree: We searched for 2 possible values of criterion (``gini'' and ``entropy''), four values of max\_depth (3, 5, 7 and 10), four values of min\_samples\_leaf (1, 2, 5 and 10) and two values of class\_weight (None, and ``balanced'').



## Visualization of the results

We created a simple dashboard to visualize all the explanations computed with our approach. You can find the code in the `UI` folder, the dashboard is also available here: [FIRE360 Dashboard](https://heroic-dasik-a43ca2.netlify.app/).
