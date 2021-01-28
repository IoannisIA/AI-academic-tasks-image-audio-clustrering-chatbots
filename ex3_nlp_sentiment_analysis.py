from textblob import TextBlob
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import pandas as pd

y = []
predicted_y = []

with open('twitter-2016test-A-clean.tsv') as tweets:
    line = tweets.readline()
    while line:
        line_parts = line.split('	')
        blob = TextBlob(line_parts[2])
        if blob.sentiment.polarity < - 0.25:
            predicted_label = 'negative'
        elif - 0.25 <= blob.sentiment.polarity < 0.25:
            predicted_label = 'neutral'
        else:
            predicted_label = 'positive'
        #print(blob.sentiment.polarity, line_parts[1], predicted_label)
        y.append(line_parts[1].strip())
        predicted_y.append(predicted_label)
        line = tweets.readline()


df = pd.DataFrame(list(zip(y, predicted_y)), columns=['y', 'predicted_y'])

print('confusion_matrix', confusion_matrix(df.y.values, df.predicted_y.values))

print('accuracy', accuracy_score(df.y.values, df.predicted_y.values))

print('recall', recall_score(df.y.values, df.predicted_y.values, average='weighted'))

print('precision', precision_score(df.y.values, df.predicted_y.values, average='weighted'))

print('f1_score', f1_score(df.y.values, df.predicted_y.values, average='weighted'))
