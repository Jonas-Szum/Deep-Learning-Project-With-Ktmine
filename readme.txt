The goal of this project with Ktmine was to create a more accurate transaction
classifier than was currently in use. All .py files were ran with python3.

The folders "Data" and "paragraphID" were provided to me for this project.
The bottom section of deepClassifier.py has been redacted because it
contained sensitive information.


The "Data" folder corresponds to my main assignment: classifying documents
with up to seventy-seven labels. To achieve this, I ended up using a
Convolutional Neural Network with a tunable false-positive/false-negative
variable. When the variable is set to 0.5, the classifications are roughly 95%
correct with an even mix of false positives and false negatives. Set the
variable below 0.5 to decrease false-negatives, and above 0.5 to decrease
false-positives.

To use:
First, run fileMaker.py to create the "parsedData" folder that deepClassifier.py
will use. On github, the parsedData folder has already been generated.
Next, run deepClassifier.py to train on the sample corpus. The final accuracy
is the accuracy which is reported after 5 epochs (~95%). 

Please note, in practice, you would only want to run deepClassifier.py once on a
much larger corpus and save the assigned weights. This program would not be run
every time a new file was introduced to the system. Rather, the saved weights 
would be used to quickly classify the new file. It is recommended to run
deepClassifier.py when a significant amount of new files have been added into the
system to prevent over-generalization of future classifications.


The "paragraphID" folder corresponds to my side project: classifying paragraphs
as payment agreements or record of right agreements. For this I used a TF-IDF
vectorizer and an MLP classifier.

To use:
Simply run paraID.py.