# 基于自适应隐私预算分配的 DP-FL 研究

## 快速开始

### 1. 激活 conda 环境
```bash
conda activate env1
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 设置环境变量（必须，Windows OMP 冲突）
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 4. 运行项目（按顺序）

#### Step 1: 验证 Flower API
```bash
python step1_verify/test_flower.py
```
**预期输出：** `✓ Flower API 验证通过！`

#### Step 2: 基线 DP-FL（均匀 ε 分配）
```bash
python step2_baseline/run_baseline.py
```

#### Step 3: 自适应 DP-FL（核心创新）
```bash
python step3_adaptive/run_adaptive.py
```

#### Step 4: 完整实验
```bash
python step4_experiments/exp1_comparison.py
python step4_experiments/exp2_ablation.py
python step4_experiments/exp3_mia.py
```

#### Step 5: 生成图表
```bash
python analysis/plot.py
```

---

## 项目结构

- `step1_verify/` - 环境验证
- `step2_baseline/` - 基线 DP-FL（均匀 ε 分配）
- `step3_adaptive/` - 自适应分配（核心创新）
- `step4_experiments/` - 完整实验
- `results/` - 实验结果输出

---

## 核心创新

**自适应隐私预算分配：**
- 根据客户端数据质量（熵 + 多样性）动态分配 ε
- 高质量数据 → 多用隐私预算（加较少噪声）
- 低质量数据 → 少用隐私预算（加较多噪声）
- 在相同总隐私成本下，提升模型准确率
