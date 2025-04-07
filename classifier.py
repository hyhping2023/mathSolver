from FlagEmbedding import FlagAutoModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import numpy as np
import os, glob
import logging
import pickle

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
    ]
)

class QueryClassifier:
    def __init__(self, model_path="/data/hyhping/BAAI/bge-m3", svm_path="outputs/model", 
                 model_dir="./outputs/model", device:list=["cuda:4"]):
        self.model = FlagAutoModel.from_finetuned(model_path,
                                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                    use_fp16=True,
                                    devices=device,)
        if os.path.exists(os.path.join(svm_path, "linear.pkl")):
            self.SVM = pickle.load(open(os.path.join(svm_path, "linear.pkl"), "rb"))
        else:
            logging.warning(f"SVM not found at {svm_path}. Please train the model first.")
            self.SVM = None
        logging.info(f"Model loaded from {model_path} on device {device}")
        
    def encode(self, query, batch_size=1024):
        # Convert the query to a vector using the model
        return self.model.encode(query, convert_to_numpy=True, batch_size=batch_size)

    def train(self, data_path="data", model_dir="./outputs/model"):
        # Implement your training logic here
        # This is a placeholder for the actual training code
        
        # from thundersvm import SVC
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        clfs = [
            # SVC(C=15, kernel='rbf', gamma=10),
            SVC(C=0.1, kernel='linear'),
            # SVC(C=0.1, kernel='poly', degree=5)
        ]
        train_data, test_data, train_labels, test_labels = self._load_data(data_path)
        for clf in clfs:
            clf.fit(train_data, train_labels)
            score = clf.score(test_data, test_labels)
            logging.info(f"Model {clf} score: {score}")
            # 预测
            y_pred = clf.predict(test_data)
            # 评估模型
            print(f"模型类型: {clf.kernel}")
            print(f"准确率: {metrics.accuracy_score(test_labels, y_pred):.4f}")
            print(f"分类报告:\n{metrics.classification_report(test_labels, y_pred)}")
            print(f"混淆矩阵:\n{metrics.confusion_matrix(test_labels, y_pred)}\n")
            # 保存模型
            pickle.dump(clf, open(os.path.join(model_dir, f"{clf.kernel}.pkl"), "wb"))
            logging.info(f"Model saved to {os.path.join(model_dir, f'{clf.kernel}.pkl')}")
        logging.info("Training completed.")

    def __call__(self, query):
        return self._predict(query)

    def _predict(self, query):
        query_vector = self.encode([query])['dense_vecs']
        # 使用SVM进行预测
        if self.SVM is not None:
            prediction = self.SVM.predict(query_vector)
            if prediction == 0:
                return "search"
            elif prediction == 1:
                return "math"
            else:
                logging.warning("Unknown prediction. Please check the model.")
                return None
        else:
            logging.warning("SVM model not found. Please train the model first.")
            return None


    def _load_data(self, data_path="data", partly=True):
        # Load your data from the specified path
        # This is a placeholder for the actual data loading code
        files = glob.glob(os.path.join(data_path, "*"))
        train_0, train_1, test_0, test_1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        test = pd.DataFrame()
        for file in files:
            if file.endswith(".parquet"):
                data = pd.read_parquet(file)['question'].to_frame()
            else:
                logging.warning(f"Unsupported file format: {file}")
                continue
            if "train" in file:
                train_0 = pd.concat([train_0, data], ignore_index=True)
            elif "test" in file:
                test_0 = pd.concat([test_0, data], ignore_index=True)
            elif "val" in file:
                test_0 = pd.concat([test_0, data], ignore_index=True)
            else:
                logging.info(f"Automatically split the data into train and test sets: {file}")
                train_data = data.sample(frac=0.8, random_state=42)
                train_1 = pd.concat([train_1, train_data], ignore_index=True)
                data = data.drop(train_data.index)
                test_1 = pd.concat([test_1, data], ignore_index=True)
        logging.info(f"训练集数量: {len(train_0) + len(train_1)}")
        logging.info(f"测试集数量: {len(test_0) + len(test_1)}")
        # 对train，test进行encoding
        # 对较大的数据进行采样
        if partly:
            length_0 = len(train_0)
            length_1 = len(train_1)
            if length_0 >= length_1:
                train_0 = train_0.sample(length_1, random_state=42)
            else:
                train_1 = train_1.sample(length_0, random_state=42)
            # length_0 = len(test_0)
            # length_1 = len(test_1)
            # if length_0 >= length_1:
            #     test_0 = test_0.sample(length_1, random_state=42)
            # else:
            #     test_1 = test_1.sample(length_0, random_state=42)
        # 将pd.DataFrame转换为list
        train_0_list = train_0['question'].tolist()
        test_0_list = test_0['question'].tolist()
        train_1_list = train_1['question'].tolist()
        test_1_list = test_1['question'].tolist()
        # 批量encode
        batch_size = 1024 # consume 20G graphic memory
        train_0 = self.encode(train_0_list, batch_size=batch_size)['dense_vecs']
        test_0 = self.encode(test_0_list, batch_size=batch_size)['dense_vecs']
        train_1 = self.encode(train_1_list, batch_size=batch_size)['dense_vecs']
        test_1 = self.encode(test_1_list, batch_size=batch_size)['dense_vecs']
        # 对数据进行标注
        # 0是search， 1是math
        label0 = np.zeros(len(train_0))
        label1 = np.ones(len(train_1))
        target0 = np.zeros(len(test_0))
        target1 = np.ones(len(test_1))
        # 合并数据
        train = np.concatenate([train_0, train_1], axis=0)
        test = np.concatenate([test_0, test_1], axis=0)
        label = np.concatenate([label0, label1], axis=0)
        target = np.concatenate([target0, target1], axis=0)
        print(train.shape)
        print(test.shape)
        return train, test, label, target

if __name__ == "__main__":
    q = QueryClassifier()
    # q.train()
    print(q("How to solve the integral of x^2 from 0 to 1?"))