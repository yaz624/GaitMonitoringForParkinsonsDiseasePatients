from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib import cm

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None)
        data = data.dropna(how='all')  # 删除包含 NaN 的行
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def preprocess_data(data):
    # 提取特征和标签
    X = data.iloc[:, 4:-1].values  # 使用第 4 列到倒数第 2 列作为特征
    y = data.iloc[:, -1].values    # 使用最后一列作为标签

    # 只保留标签为 0 和 1 的行
    data = data[(y == 0) | (y == 1)]
    X = data.iloc[:, 4:-1].values
    y = data.iloc[:, -1].values

    # 确保标签为整数类型
    y = y.astype(int)

    return X, y

def train_svm(X_train, y_train):
    # 训练 SVM 模型
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train, y_train)
    return model

def plot_3d_decision_boundary(X, y, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='winter', s=30)

    # 创建网格
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx, yy, zz = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 20),
        np.linspace(ylim[0], ylim[1], 20),
        np.linspace(zlim[0], zlim[1], 20)
    )

    # 将网格点展开为二维点集
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    decision_values = model.decision_function(grid)

    # 绘制 3D 决策边界
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=decision_values, cmap='coolwarm', alpha=0.2, s=1)

    # 设置标题和轴标签
    ax.set_title("3D SVM Decision Boundary")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")

    # 添加颜色条
    norm = plt.Normalize(vmin=decision_values.min(), vmax=decision_values.max())
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax, shrink=0.5, aspect=5)

    plt.show()

def main():
    # 加载数据
    data = load_data('cleaned_data.txt')
    if data is None:
        return

    # 预处理数据
    X, y = preprocess_data(data)

    # 检查数据
    print("数据标签唯一值:", np.unique(y))

    # 确保训练集和测试集类别分布一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 检查训练集是否有足够类别
    if len(np.unique(y_train)) < 2:
        raise ValueError("训练集需要至少包含两个类别！")

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用 PCA 将数据降维到 3 维
    pca_3d = PCA(n_components=3)
    X_train_3d = pca_3d.fit_transform(X_train)
    X_test_3d = pca_3d.transform(X_test)

    # 打印降维后数据形状
    print("降维后训练集的形状（3维）：", X_train_3d.shape)

    # 训练 SVM 模型
    model_3d = train_svm(X_train_3d, y_train)

    # 模型测试
    accuracy_3d = model_3d.score(X_test_3d, y_test)
    print("降维到 3 维的测试集准确率：", accuracy_3d)

    # 可视化 3 维数据和决策边界
    plot_3d_decision_boundary(X_train_3d, y_train, model_3d)
    print("SVM model support vectors:", model.support_vectors_)
plt.show()
if __name__ == "__main__":
    main()