import pandas
import sklearn.metrics as metrics

df = pandas.read_csv('classification.csv')

clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.items():
    clf_table[name] = len(df[(df['true'] == res[0]) & (df['pred'] == res[1])])

print(clf_table)

print(metrics.accuracy_score(df['true'], df['pred']))

print(metrics.precision_score(df['true'], df['pred']))

print(metrics.recall_score(df['true'], df['pred']))

print(metrics.f1_score(df['true'], df['pred']))

df2 = pandas.read_csv('scores.csv')

scores = {}
for clf in df2.columns[1:]:
    scores[clf] = metrics.roc_auc_score(df2['true'], df2[clf])

print(scores)

pr_scores = {}
for clf in df2.columns[1:]:
    pr_curve = metrics.precision_recall_curve(df2['true'], df2[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

print(pr_scores)