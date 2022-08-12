import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models-scores.csv", delimiter="\t")

model1, model2, model3, model4, avg1, avg2, avg3, avg4, X = [], [], [], [], [], [], [], [], []

for column in df.columns.drop("Avg"):
    X.append(column)
    model1.append(df[column][0])
    model2.append(df[column][1])
    model3.append(df[column][2])
    model4.append(df[column][3])
    avg1.append(df["Avg"][0])
    avg2.append(df["Avg"][1])
    avg3.append(df["Avg"][2])
    avg4.append(df["Avg"][3])

print(model1, model2, model3, model4)
print(X)

everyAvg = (avg1[0] + avg2[0] + avg3[0] + avg4[0]) / 4
avg = []
for _ in range(200):
    avg.append(everyAvg)


plt.scatter(X, model1, color="orange", marker="o", s=8, label="model1")

plt.scatter(X, model2, color="blue", marker="o", s=8, label="model2")

plt.scatter(X, model3, color="purple", marker="o", s=8, label="model3")

plt.scatter(X, model4, color="green", marker="o", s=8, label="model4")

plt.plot(X, avg, linestyle="--", color="red", linewidth=2, label="Avg")


plt.xlabel("Game")
plt.ylabel("Score")
plt.title("Each model")
plt.legend(fontsize="small")

plt.show()




