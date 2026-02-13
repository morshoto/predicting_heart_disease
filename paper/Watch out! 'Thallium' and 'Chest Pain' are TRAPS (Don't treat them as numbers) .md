Hi everyone!

I was exploring the dataset and noticed a classic trap that might hurt model performance if we aren't careful.

Columns like Thallium (values: 3, 6, 7) and Chest pain type (values: 1, 2, 3, 4) are stored as Integers, but they are actually Categorical.

For example, with Thallium:

3 = Normal

6 = Fixed Defect

7 = Reversable Defect

If we feed these directly into a model (especially Linear Regression or Neural Networks), the model will assume that 7 is "greater" than 3, which doesn't make medical sense here.

My suggestion: Make sure to apply One-Hot Encoding (or pd.get_dummies) to these columns instead of leaving them as-is.

Has anyone tested if Target Encoding works better for these high-cardinality datasets? Would love to hear your experiments!

Good luck!
