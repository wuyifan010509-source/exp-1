"""
实验配置文件 - 请填写您的配置
"""
import os

# ============ GPU模型配置 ============
# 在您的GPU服务器上部署的模型
BACKBONE_MODEL = "Qwen2.5-32B-Instruct"  # 或其他模型名称
# BACKBONE_API_URL = "http://172.17.160.46:8080/v1"  # 修改为您的GPU服务器地址
BACKBONE_API_URL="http://127.0.0.1:3002/v1"

# ============ API配置 ============
# DeepSeek API用于白盒初始化和变异
DEEPSEEK_API_KEY = "sk-bf6974bd1ad94090acf36449518b8417"  # 填入您的DeepSeek API Key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ============ 遗传算法超参数 ============
POPULATION_SIZE = 20
N_GENERATIONS = 30
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3
ELITE_COUNT = 2
TOURNAMENT_K = 3
LAMBDA_PENALTY = 100
MAX_PROFILE_LENGTH = 200

# 槽位字数限制
MAX_C_LENGTH = 80
MAX_B_LENGTH = 20  # 处理边界限制20字
MAX_R_LENGTH = 20  # 拒绝范围限制20字

# ============ 数据路径 ============
GOLDEN_TEST_PATH = "data/GOLDEN_TEST.csv"
HISTORICAL_LOGS_PATH = "data/HISTORICAL_LOGS.csv"
AGENTS_PATH = "data/agents/tools_descriptions.json"
RESULTS_DIR = "results/"

# ============ 评估配置 ============
EVAL_SUBSET_SIZE = 100  # 适应度评估使用的子集大小（加速）
FULL_EVAL_SIZE = 354    # 完整测试集大小
MARGIN_THRESHOLD = 0.1
MARGIN_SAMPLING_INTERVAL = 5
MARGIN_TOP_K = 20

# ============ 最小实验配置 ============
MVE_POP_SIZE = 5
MVE_GENERATIONS = 5
MVE_EVAL_SUBSET = 20

# 12个意图类别（去除oos）
INTENT_CLASSES = [
    "选股类",
    "诊股类", 
    "预测类",
    "知识库类",
    "新闻类",
    "通用类",
    "推荐类",
    "策略类",
    "指标查询类",
    "身份类",
    "分时图类",
    "K线图类"
]

# 意图到智能体名称的映射
INTENT_TO_AGENT = {
    "选股类": "选股智能体",
    "诊股类": "诊股智能体",
    "预测类": "预测智能体", 
    "知识库类": "知识库智能体",
    "新闻类": "新闻智能体",
    "通用类": "通用智能体",
    "推荐类": "推荐智能体",
    "策略类": "策略智能体",
    "指标查询类": "指标查询智能体",
    "身份类": "身份智能体",
    "分时图类": "分时图智能体",
    "K线图类": "K线图智能体"
}
