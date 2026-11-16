# FairLens AI: Credit-Default Model Comparison and Group Fairness Audit

**Track:** AI + Finance | **Team:** Solo project Elton Wedam| **Dataset:** UCI Default of Credit Card Clients (30,000 records) | **Code:** https://colab.research.google.com/drive/1SC-IBVNY6c6yjJfiYXN97kxRwejmt0uX?usp=sharing

## The problem

Lenders and risk analysts use models to predict which customers may default, but a model can perform reasonably well overall while still making different kinds of errors for different groups. FairLens helps an analyst see model performance and group fairness together, so the trade-offs are easier to understand before a model is considered for use.

## What I built

FairLens is an educational machine-learning project that compares credit-default prediction models and audits their errors across the dataset's recorded SEX codes (1 and 2). The intended user is a risk analyst at a small lender who reviews a model's group-level errors before deciding whether to use it. The input is a customer's credit and payment history (23 features). The output is a default-risk prediction plus a report of default recall, false-positive rate and selection rate for each group, shown in a Gradio dashboard in the notebook.

I compared Logistic Regression, Random Forest, Gradient Boosting and Decision Tree models, and tested a Fairlearn Equalized Odds approach. The data was split 60% training, 20% validation and 20% test. SEX was excluded from the model and kept only for evaluation. Decision thresholds were chosen on validation data, and all reported results use the held-out test split.

## What I added

I started from suggested project 4.2.3 (fairness audit of a loan model). I added a different dataset (UCI Default of Credit Card Clients), a comparison of four model types, a Fairlearn mitigation with its accuracy cost, a proxy check, and analysis of real mistakes.

## Results

A baseline that always predicts "no default" is 77.9% accurate but catches 0.0% of defaulters.

- **Random Forest:** 76.8% accuracy, 71.8% balanced accuracy, 62.8% default recall, 48.1% precision, 19.2% false-positive rate.
- **Gradient Boosting:** highest default recall (73.0%), but also the highest false-positive rate (29.6%) and 41.2% precision.
- **Group audit (Random Forest, SEX codes 1 vs 2):** default recall 64.3% vs 61.6% (gap 2.8 points); false-positive rate 20.4% vs 18.5% (gap 1.8 points).
- **Fairlearn Equalized Odds (seed 42, decision tree):** the combined group gap (the larger of the recall and false-positive-rate gaps) decreased from 3.2 to 1.9 points. However, balanced accuracy fell from 71.3% to 68.5% and the overall false-positive rate rose from 16.5% to 26.4%. This shows that reducing one fairness gap can come with a performance trade-off. It is one seeded run on one split, not proof of fairness.
- **Proxy check:** a model predicting the SEX code from the remaining features reached an AUC of 0.645 (0.50 means no signal), mostly using AGE and LIMIT_BAL. Removing SEX does not fully remove its signal.
- **Feature importance:** PAY_0 (last month's repayment status) was highest, at 0.066. This shows what the model uses, not what causes default.

**Where it fails:** the Random Forest missed 494 of the 1,327 defaulters in the test set (37%) and wrongly flagged 899 of 4,673 non-defaulters. One missed defaulter had no payment delay (PAY_0 = 0), a 350,000 credit limit and a 68,940 payment against an 82,964 bill, so the model gave only a 6% default probability. One false alarm was two months late with a balance of 1,050 and got a 96% probability. I think the model relies heavily on recent repayment status and misses customers whose behavior differs.

## Limitations and next steps

This analysis uses one historical dataset and one train/validation/test split. The two recorded SEX codes are not complete demographic identities. The results do not establish that a model is legal or suitable for lending. With more time I would test additional datasets, evaluate more groups and fairness measures, check whether results hold across different random splits, and test ways to reduce the proxy signal from AGE and LIMIT_BAL.

This project is a prototype and is not financial advice.

## Credits and resources

- **Dataset:** Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository, University of California, Irvine. https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients
- **Fairness framework:** Fairlearn (ExponentiatedGradient, EqualizedOdds). https://fairlearn.org/main/user_guide/mitigation/reductions.html
- **User interface:** Gradio, used for the results dashboard. https://gradio.app/guides/quickstart
- **Libraries:** Python, scikit-learn, pandas, NumPy, ucimlrepo
- **Development environment:** Google Colab
- **AI assistance:** ChatGPT (OpenAI) and Claude (Anthropic), used for coding help, explanations and project development
