# FIRE360

A comprehensive and fast local explanation approach tailored for tabular data.

## How to use

You can find an example of how to use FIRE360 in the `example` folder. 

## Dataset Pre-Processing 


We evaluated \fire on six datasets: Adult, Dutch, Covertype, House16, Letter, and Shuttle.
For each dataset, we performed the following pre-processing: 
* Adult: we encoded the categorical variables using OneHot encoding, and then we applied a MinMaxScaler.
* Dutch is a numeric dataset, therefore, we only used a MinMaxScaler.
* For Letter, since all the variables were numerical, we only used a MinMaxScaler.
* For Shuttle, since all the variables were numerical, we only used a MinMaxScaler.
* House16, since all the variables were numerical, we only used a MinMaxScaler.
* For Covertype, we converted the categorical variables using OneHot encoding, and then we applied a MinMaxScaler.

## Visualization of the results

We created a simple dashboard to visualize all the explanations computed with our approach. You can find the code in the `UI` folder, the dashboard is also available here: [FIRE360 Dashboard](https://heroic-dasik-a43ca2.netlify.app/).
