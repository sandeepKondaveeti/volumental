# volumental Solution & Follow up questions with respective answers.

Solution #1: Apllied just Multioutput Random Forest Regressor to Predict all the desired values.

Execution Steps for above solution: Just Navigate to the Folder
Run the Python file by Python "Filename.py"

Solution #2: Applied multiple algorithms on each and every desired output before predicting the values the function tries applying all the models and on basis of Low RMSE Values, At present we have choosen Linear Regression which have a low RMSE Values and then Started predicting the results and the outputs are stored in the same location in the file " submit_file.csv".

Execution Steps for above solution: Just Navigate to the Folder
Run the Python file by Python "Filename.py"


Follow up questions given in the doc and the answers respectively.

1. Imagine that your solution works well and a decision is made to make it into a product and take it into production as a service. What steps would you take and what would you improve or change in preparation for that? How would you deploy the model?

A: For a very basic deployment, assuming We have our code in Python, We will just need to create a Flask app, define a function that contains the code for our model and returns the output.

For more sophisticated deployments, we can look at Docker to create the Flask/Python images and deploy them on a Docker container. We can also look to leverage Serverless platforms like AWS Lambda can be used to scale up the servers instantly. We can make use of AWS/Azure for cloud storage, CI/CD.

Using python Flask or Django web framework for making rest API, which takes features as a input from web users and predicts the output according to the machine learning algorithm.

Avoiding overfitting by Regularization(L1/L2), Scaling to work on multi-cores, Empowering the functions written so that we can get the best outcome of the data provided by the users.

2. Suppose that a team will continue working on this solution continuously going forward. What do you believe would be important steps to take to make that team effective?

A: Avoiding overfitting by Regularization(L1/L2), Scaling to work on multi-cores,  Dockers for resolving environmental issues and making it into GIT which can help the team for continuous integration. 

Working in Agile methodology will help in continuous delivery, Regular adaptation to changing circumstances and Even late changes in requirements are welcomed.
