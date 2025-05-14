This project uses a support vector classifier on a dataset containing people's mock bank information (https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
to predict if they would be accepted for a loan. The results demonstrate the misleading nature of false positives/negatives. Although the accuracy of the model
was ~83%, recall was ~35%, precision was ~70, specificity was ~95%. With such a low recall and a view of the confusion matrix, we can see that the model is mainly predecting
a "rejected" loan status. 

![image](https://github.com/user-attachments/assets/b3a94d3c-5872-416b-8df7-fb2b2a99588e)

This is because, after review, most of the loan statuses are rejected. Therefore, the model is getting used to a variety of feature values being associated
with the "rejected" status, causing the model to usually predict "rejected." Since the the dataset is mostly "rejected" loans, the model is typically correct, but will
predict inacurrately when assessing a loan that has a true value of "accepted."
